"""Generate the two narrative notebooks for Study 856 (Book-Tax Differences).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR annual pretax / tax /
# assets + yfinance prices, 39 large-cap names, fiscal-year ends 2007-12 -> 2025-12, as-of
# 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=39, n_events=569,
    end_lo="2007-12-31", end_hi="2025-12-31", fp_prices="d579ef33a57f",
    share_pos_btd=71, btd_med=2.1,
    # primary calendar long-short (btd_neg, terciles, staleness 430) — long low-BTD, short high-BTD
    n_months=196, avg_n=29.9, ls_span_lo="2010-03-31", ls_span_hi="2026-06-30",
    ls_mean_bps=20.1, ls_ann=2.42, ls_t_iid=0.75, ls_t_nw=0.74, ls_sharpe=0.19,
    ls_hit=49, ls_long_bps=151.9, ls_short_bps=131.8, ls_turn=0.05, ls_cum=1.29,
    change_mean_bps=37.9, change_t_nw=1.79, change_sharpe=0.47,
    xsec_early=6, xsec_late=37,
    # pooled event drift  horizon -> (n, top(low-BTD)%, bot(high-BTD)%, ls%, t, win%, placebo p)
    drift={
        63: (569, 2.80, 3.00, -0.20, -0.15, 47, 0.5674),
        126: (551, 4.95, 7.42, -2.47, -1.43, 49, 0.9369),
        252: (541, 15.73, 17.35, -1.62, -0.55, 49, 0.7285),
    },
    # terciles by btd_neg low->high == BTD high->low  (biggest gap -> smallest gap), 252d
    mono252=(17.35, 16.22, 15.73),
    # era split (calendar LS), split at 2018 (TCJA)
    era_early_n=94, era_early_bps=29.2, era_early_t=0.94,
    era_late_n=102, era_late_bps=11.8, era_late_t=0.27,
    # earnings persistence (Hanlon mechanism)
    pers_n=530, b_all=0.750, b_low=0.750, b_high=0.752, pers_diff=0.002, pers_t=0.03, pers_r2=0.570,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (15.0, 1.80, 0.55, 0.14), (20, 100): (9.8, 1.18, 0.36, 0.09)},
    # synthetic control
    syn_null_mean=0.31, syn_null_sd=0.67, syn_null_fire=0, syn_null_seeds=12,
    syn_planted_bps=62.8, syn_planted_t=2.55,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Persistence%3F: Not detected](https://img.shields.io/badge/Persistence%3F-Not_detected-8b949e?style=flat-square)\n\n"
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

from book_tax_diff import data, strategy as st

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
            "# A tax-return red flag for stocks — that blue chips shrug off 🧮\n"
            "### When a company's book profit towers over the profit its tax bill implies, is that "
            "a warning? On 39 giants, it warns of *nothing*.\n\n"
            + BADGES +
            "Every company keeps two sets of books — legally. One follows accounting rules (the "
            "**book income** you see on the income statement); the other follows the tax code (the "
            "**taxable income** the IRS actually taxes). They almost never match. When book income "
            "sits *far above* the taxable income its tax bill implies — a big **book-tax "
            "difference** — a famous accounting paper (Hanlon 2005) says be careful: those "
            "earnings are being propped up by things that haven't hit the tax return, so they're "
            "**less likely to last**, and the stock should do worse.\n\n"
            "Great story. So we ranked 39 big US companies on exactly that gap and checked both "
            "halves of the claim — the returns *and* the fragile-earnings part. Neither one shows "
            "up.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "persistence regression? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 42 large US filers (39 with usable history), annual 10-K "
            "figures 2007→2026; a **thin panel skewed to the wrong tail** — these blue chips are "
            "where book-tax gaps are *smallest*. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a big book-tax gap predict **lower returns**? | **Not in any way we can "
            f"certify.** A long-short that owns the low-gap names and shorts the high-gap ones is "
            f"*right-signed* (+{R['ls_ann']:.1f}%/yr on paper) but statistically nothing (robust "
            f"*t* = **+{R['ls_t_nw']:.2f}**, the bar is 2) — and a second way of measuring it "
            f"actually leans the **opposite** way. |\n"
            f"| Does it mark **fragile (less-persistent) earnings**? | **No.** That's Hanlon's "
            f"actual headline, and here it's flat: high-gap firms' earnings persist "
            f"**{R['b_high']:.3f}** vs low-gap firms' **{R['b_low']:.3f}** — the same number. |\n"
            "| Why the nothing? | We're looking in the wrong place *on purpose*: big, clean, "
            "heavily-audited blue chips are where these gaps are smallest and best-explained. The "
            "red flag lives in the messy tail — small, distressed, shelter-heavy names — none of "
            "which are in a large-cap survivor basket. |\n"
            "| Can you trade it? | **No.** The return edge doesn't clear the bar even before "
            "costs. |\n\n"
            "> A red flag that, waved at blue chips, flags nothing — neither the stock nor the "
            "earnings behind it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Book income far above the income the tax bill implies means the earnings are "
            "inflated by soft, reversing items. Short the big-gap 'aggressive' names, own the "
            "low-gap 'clean' ones.\"*\n\n"
            "It's a specific, tax-flavoured case of a respectable academic idea — the market "
            "misprices the *accrual* pieces of earnings (Sloan 1996). The book-tax gap is an "
            "unusually clean read on one accrual: the stuff accounting counts as profit that the "
            "tax code doesn't (yet). Hanlon (2005) showed such firms have **less persistent "
            "earnings**. The question is whether that's *tradeable* — and whether it even holds "
            "among household-name companies."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If one number from the tax footnote flagged which stocks would lag, it would be a "
            "gift — it's in every 10-K, no estimates feed, no alt-data. That's exactly why it "
            "deserves suspicion: it's *too* readable. And there's a second reason to doubt it here "
            f"— **{R['share_pos_btd']}%** of these firm-years already show book income *above* the "
            "tax-implied figure (median gap ≈"
            f"{R['btd_med']:.0f}% of assets). A 'positive book-tax difference' is the normal state "
            "of a profitable company, not an exotic warning sign. So 'large' has to be doing a lot "
            "of work — and on blue chips, it isn't."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** For each of {R['n_names']} big US companies, the book-tax gap = "
            "(pretax book income − the taxable income its tax bill implies) ÷ assets, from the "
            "annual 10-K, known only on the **filing date** (no peeking at a number before it's "
            "public). We gross up the tax bill by the **statutory rate** — 35% before 2018, 21% "
            "after the tax-cut law.\n"
            "- **The return test.** Each month, rank the names on the gap, buy the low-gap third, "
            "short the high-gap third, hold one month. Does the spread make money — and can we "
            "tell it apart from noise?\n"
            "- **The earnings test.** Do the high-gap names' profits actually persist less next "
            "year? (Hanlon's real claim.)\n"
            "- **The mirage check.** If the return spread can't beat a coin-flip relabelling of "
            "the names, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Start with the mechanism, because it's the cleanest test.** Hanlon says high-gap "
            "earnings are *fragile* — this year's profit is a worse guide to next year's. So we "
            "measure how well each group's ROA carries over to the following year, and compare the "
            "biggest-gap third to the smallest-gap third."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.earnings_persistence(EV)\n"
            "    b_low, b_high = q['b_low'], q['b_high']\n"
            "else:\n"
            "    b_low, b_high = R['b_low'], R['b_high']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['smallest-gap third\\n(\"clean\")','biggest-gap third\\n(\"aggressive\")'],\n"
            "       [b_low, b_high], color=[GREEN, GREY], width=.55)\n"
            "for i,v in enumerate([b_low, b_high]): ax.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('earnings persistence (next-year ROA vs this-year)')\n"
            "ax.set_ylim(0, 1.0)\n"
            "ax.set_title('Hanlon says these bars should differ. They do not.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'earnings persistence: clean {b_low:.3f} vs aggressive {b_high:.3f} '\n"
            "      f'-> gap {b_high-b_low:+.3f} (should be clearly negative if the claim held)')"
        ),
        md(
            f"The two bars are **the same height** — persistence {R['b_low']:.3f} for the "
            f"clean third, {R['b_high']:.3f} for the aggressive third, a gap of "
            f"**{R['pers_diff']:+.3f}** (statistically zero). On these names, a big book-tax "
            "difference does **not** mark fragile earnings. The engine the whole claim runs on "
            "isn't turning over.\n\n"
            "**So does the stock at least move?** Same ranking, but now we measure the forward "
            "*return* of a buy-the-low-gap, short-the-high-gap portfolio, rebalanced monthly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='btd_neg', n_buckets=3, min_names=6, staleness_days=430)\n"
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
            "    ax.set_title(f'A gently rising line that means nothing: +{ann:.1f}%/yr but robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: +{ann:.1f}%/yr gross, but Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"It *drifts up* — **+{R['ls_ann']:.1f}%/yr** gross, growing \\$1 to "
            f"\\${R['ls_cum']:.2f} over {R['n_months']} months, and it's right-signed (clean beats "
            f"aggressive, as Hanlon would want). But that's the trap. Once you account for how "
            f"noisy those monthly returns are, the robust *t* is **+{R['ls_t_nw']:.2f}** — nowhere "
            f"near the **2** we require. A line that slopes up and a *reliable* edge are different "
            "things.\n\n"
            "**And here's the tell that kills it:** measure the same names a second way — the raw "
            "drift after each filing, sorted by gap size — and the ranking *flips*. The biggest-gap "
            "names, the ones you're supposed to short, actually drift *up* a touch more."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=252)\n"
            "    mono = st.bucket_means(fr, 3)*100     # buckets of btd_neg low->high == BTD high->low\n"
            "else:\n"
            "    mono = np.array(R['mono252'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['biggest\\ngap','middle','smallest\\ngap'], mono,\n"
            "       color=[GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 1-year forward return')\n"
            "ax.set_title('The wrong ladder: biggest-gap names drift up a touch MORE (anti-Hanlon, but tiny)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1-year forward return by gap size (biggest->smallest):', [f'{v:+.1f}%' for v in mono])"
        ),
        md(
            f"Sorted by gap size, the forward returns run **{R['mono252'][0]:+.1f}% / "
            f"{R['mono252'][1]:+.1f}% / {R['mono252'][2]:+.1f}%** from biggest to smallest gap — a "
            "faint slope in the **opposite** direction to the claim, and far too small to be real. "
            "One way of measuring says clean-beats-aggressive; the other says the reverse; both are "
            "indistinguishable from zero. That's what *no signal* looks like: the sign isn't even "
            "stable."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Fragile earnings? — Not detected.** The mechanism Hanlon documented is flat "
            f"here: persistence {R['b_low']:.3f} (clean) vs {R['b_high']:.3f} (aggressive). No "
            "difference.\n"
            f"- **Signal (returns) — Weak.** The long-short is right-signed and ~"
            f"+{R['ls_ann']:.1f}%/yr on paper, but never clears the bar (robust *t* = "
            f"+{R['ls_t_nw']:.2f}), the second measurement leans the other way, and neither era "
            "certifies. A whiff of the story, nothing this tape can stand behind.\n"
            "- **Tradability — Mirage.** You can't get paid for an edge you can't distinguish from "
            "luck. It fails before costs are even charged.\n\n"
            "> The honest one-liner: *the book-tax red flag may well work in the messy corners of "
            "the market — but on blue chips it flags neither the earnings nor the stock.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where the effect probably still lives.** Not in mega-caps but in the **broad "
            "cross-section** — small, distressed, tax-shelter-heavy names where book and tax income "
            "genuinely part ways and few analysts are watching. A survivor basket of giants is the "
            "*hardest* place for this signal, by construction. The null here refutes the *blue-chip* "
            "version, not Hanlon's.\n"
            "- **The coverage caveat is real.** The cross-section runs from ≈"
            f"{R['xsec_early']:.0f} names early to ≈{R['xsec_late']:.0f}, and three names dropped "
            "for want of a clean pretax-income tag. A wider, deeper, *non-survivor* panel is the "
            "honest next step.\n"
            "- **Sibling studies:** [568-effective-tax-rate](../../568-effective-tax-rate/) (the "
            "tax *rate* level, not the book-tax *gap*), [231-sloan-accruals](../../231-sloan-accruals/) "
            "(the broad accruals anomaly BTD is a slice of), and "
            "[229-beneish-m-score](../../229-beneish-m-score/) (an eight-signal manipulation score). "
            "See [docs/references.md](../docs/references.md) for the exact dedup.\n\n"
            "*Think the flag works where the messy firms are? Build the wide, survivor-free panel, "
            "show a certifiable persistence gap and a net return spread on the size you'd actually "
            "run — then we'll talk.*"
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
            "# Book-Tax Differences — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a pooled "
            "event-drift cross-check with a label-shuffle placebo · an era split · the "
            "earnings-persistence interaction · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **a large positive book-tax difference marks less-persistent earnings and "
            "predicts lower returns** (Hanlon 2005) — splits into a *mechanism* piece and a "
            "*mispricing* piece, and on this large-cap survivor tape **both come back null**. This "
            "is distinct from every sibling on the desk: [568](../../568-effective-tax-rate/) ranks "
            "on the tax *rate*, [231](../../231-sloan-accruals/) on *total accruals*, "
            "[229](../../229-beneish-m-score/) on an *eight-signal manipulation score*. This is the "
            "statutory-grossed-up **book-minus-tax income gap**.\n\n"
            "> ⚠️ **Data note.** EDGAR annual `IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
            "...` / `IncomeTaxExpenseBenefit` / `Assets` + yfinance adjusted closes, "
            + str(R["n_names"]) + " names, ends " + R["end_lo"] + " → " + R["end_hi"]
            + ", as-of " + R["as_of"] + ". Point-in-time on the **10-K filing date**. "
            "Survivorship named on the Signal axis (current-survivors basket, skewed to the tail "
            "where the effect is *weakest*). Numbers in [`docs/results.md`](../docs/results.md) "
            "(prices fingerprint `" + R["fp_prices"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** (returns) | `WEAK` | calendar tercile long-short (long low-BTD, short "
            f"high-BTD) **{R['ls_mean_bps']:+.1f} bps/mo** (+{R['ls_ann']:.1f}%/yr gross), "
            f"one-sample *t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; "
            f"change variant NW *t* = {R['change_t_nw']:+.2f}; pooled drift mildly *anti*-Hanlon — "
            "right sign not even stable |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: NW *t* = "
            f"{R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; fails "
            "**before** costs |\n"
            f"| **Marks fragile earnings?** | `NOT DETECTED` | persistence interaction "
            f"(high-BTD − low-BTD) = {R['pers_diff']:+.3f} (*t* = {R['pers_t']:+.2f}); "
            f"{R['b_high']:.3f} vs {R['b_low']:.3f} — identical |\n\n"
            "> 💡 In plain words: on blue chips, **neither** the return spread **nor** the "
            "earnings-persistence engine behind it shows up. The mispricing needs the mechanism; "
            "the mechanism is absent here."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_{i,y}$ be name $i$'s pretax book income for fiscal year $y$ and $T_{i,y}$ its "
            "income-tax expense, disclosed on 10-K filing date $F_{i,y}$, with statutory rate "
            "$\\tau_y$ (0.35 pre-2018, 0.21 after). The **book-tax difference**, scaled by assets "
            "$A$:\n\n"
            "$$\\text{BTD}_{i,y} = \\frac{P_{i,y} - T_{i,y}/\\tau_y}{A_{i,y}}$$\n\n"
            "We gross up by the **statutory** (not effective) rate on purpose — dividing by the "
            "effective rate would algebraically recover $P$ and define the gap away. The claims:\n\n"
            "- **H₁ (persistence).** High BTD marks **less persistent** earnings: next-year ROA "
            "loads less on this-year ROA — Hanlon's mechanism.\n"
            "- **H₂ (returns).** A cross-sectional long-short, long low-BTD / short high-BTD, earns "
            "a positive forward spread — the mispricing claim.\n"
            "- **H₃ (tradable).** That spread survives realistic long-short costs + borrow.\n\n"
            "We find **H₁ not supported** (interaction "
            f"{R['pers_diff']:+.3f}, *t* = {R['pers_t']:+.2f}), **H₂ not supported** (NW *t* = "
            f"{R['ls_t_nw']:+.2f}, and the pooled drift is mildly anti-Hanlon), and therefore "
            "**H₃ moot**. The literature (Hanlon 2005) documents the effect in the broad "
            "cross-section; our large-cap survivor tape supports **neither** piece — hence `WEAK` "
            "on returns (right-signed but uncertified) and `NOT DETECTED` on the mechanism."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, because "
            "balance-sheet/annual signals are persistent and filings cluster: a calendar series of "
            "monthly long-short returns lets a **Newey-West (6-lag) HAC *t*** do the honest work "
            "the desk's `REAL` bar is written against. The panel is thin, so we sort into "
            "**terciles** (not quintiles) and require ≥ 6 names. The pooled event drift + a "
            "**label-shuffle placebo** is the cross-check; the persistence axis is an **interaction "
            "OLS** graded on the *magnitude* of the slope difference (its pooled *t* ignores "
            "firm/year clustering)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, fiscal-year) events across {R['n_names']} "
            f"names, ends {R['end_lo']} → {R['end_hi']}, each stamped with its 10-K filing date "
            "(point-in-time).\n"
            "- **Primary.** Monthly tercile long-short on `btd_neg = −BTD/Assets`, one execution "
            "lag (rank at month $M$ close, earn month $M{+}1$); Newey-West + one-sample *t*, Sharpe, "
            "hit rate.\n"
            "- **Cross-check.** Pooled event drift over 63/126/252 trading days, one-day-lag entry, "
            "top-minus-bottom tercile, one-sample *t* + 10k-draw label-shuffle placebo, and the "
            "tercile monotonicity picture.\n"
            "- **Robustness.** The year-on-year **change** in the gap `−ΔBTD/Assets`; an era split "
            "at 2018 (the TCJA statutory-rate cut).\n"
            "- **Mechanism.** Interaction OLS of next-year ROA on this-year ROA × high-BTD tercile.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short borrow.\n"
            "- **Control.** Synthetic panel, planted-effect knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh signals into terciles each month, long low-BTD / short high-BTD "
            "equal-weight, earn next month's return. The decisive statistic is the HAC *t* of the "
            "monthly long-short series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='btd_neg', n_buckets=3, min_names=6, staleness_days=430)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls_ch = st.calendar_ls(PX, EV, signal_col='d_btd_neg', n_buckets=3, min_names=6, staleness_days=430)\n"
            "    s_ch = st.calendar_ls_stats(ls_ch)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo (+{s['ann_pct']:.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  change signal (-dBTD/Assets): {s_ch['mean_bps']:+.1f} bps/mo, NW t = {s_ch['t_nw']:+.2f}, Sharpe {s_ch['sharpe']:.2f}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=AMBER, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: +{R['ls_ann']:.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven: the panel widens as XBRL history fills in')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"change-signal NW t = {R['change_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **+{R['ls_mean_bps']:.0f} bps/month** is right-signed (low-BTD "
            f"beats high-BTD, ~+{R['ls_ann']:.1f}%/yr gross), but the HAC *t* is "
            f"**+{R['ls_t_nw']:.2f}** — the monthly returns are too noisy and too few "
            f"(n={R['n_months']}, avg cross-section {R['avg_n']:.0f}) to distinguish from zero. The "
            f"strongest specification, the year-on-year **change** in the gap, only reaches NW *t* "
            f"= +{R['change_t_nw']:.2f}. Right-signed, never certified."
        ),
        md(
            "### 4b · The cross-check — pooled event drift + placebo + monotonicity\n\n"
            "Bucket all events by `btd_neg` (top = low BTD); top-minus-bottom forward drift with a "
            "label-shuffle null. If there were a sort, the terciles would form a ladder — the "
            "*right* way."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for h in st.HORIZONS:\n"
            "        es = st.event_summary(PX, EV, horizon=h, n_buckets=3, n_draws=4000)\n"
            "        rows.append((h, es['ls_mean']*100, es['t'], es['ls_win']*100, es['p_placebo']))\n"
            "    mono = st.bucket_means(st.event_drift_frame(PX, EV, horizon=252), 3)*100\n"
            "else:\n"
            "    for h in st.HORIZONS:\n"
            "        d = R['drift'][h]; rows.append((h, d[3], d[4], d[5], d[6]))\n"
            "    mono = np.array(R['mono252'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "hs = [r[0] for r in rows]; ts = [r[2] for r in rows]\n"
            "a1.bar([f'{h}d' for h in hs], ts, color=GREY, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('one-sample t (long-short drift)'); a1.set_title('Flat & sign-unstable: no horizon near |t|=2')\n"
            "a2.bar(['biggest\\ngap','middle','smallest\\ngap'], mono, color=GREY, width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('1-year forward return'); a2.set_title('Wrong-way ladder (biggest gap drifts up MORE)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift is **negative** at every horizon "
            f"({R['drift'][63][3]:+.2f}% to {R['drift'][126][3]:+.2f}%), i.e. the low-BTD names the "
            f"claim says to *own* slightly *under*-perform — the opposite of the calendar sort, and "
            f"still insignificant (*t* from {R['drift'][63][4]:+.2f} to {R['drift'][126][4]:+.2f}, "
            f"placebo *p* {R['drift'][63][6]:.2f}–{R['drift'][126][6]:.2f}). The terciles run "
            f"{R['mono252'][0]:+.1f}%/{R['mono252'][1]:+.1f}%/{R['mono252'][2]:+.1f}% "
            "biggest→smallest gap — a faint *anti*-Hanlon ladder. The calendar and event views "
            "disagree on sign and agree on insignificance: no return sort, either way."
        ),
        md(
            "### 4c · Era split — nothing hiding in a regime\n\n"
            "Split the calendar long-short at 2018 (the TCJA statutory-rate cut, 35% → 21%)."
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
            "ax.bar([f'pre-2018\\n(n={en})', f'2018-2026\\n(n={ln})'], [eb, lb], color=[AMBER, AMBER], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Both eras: right-signed, neither significant, weaker after the rate cut')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2018: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2018-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: {R['era_early_bps']:+.0f} bps (NW *t* = {R['era_early_t']:+.2f}) "
            f"pre-2018, {R['era_late_bps']:+.0f} bps (NW *t* = {R['era_late_t']:+.2f}) after — the "
            "same small right-signed-but-insignificant positive, if anything *weaker* once the "
            "statutory rate dropped. It never certified in either regime."
        ),
        md(
            "### 4d · The mechanism — do big gaps mark *fragile* earnings?\n\n"
            "The Hanlon headline, tested directly: an interaction OLS of next-year ROA on this-year "
            "ROA, with the persistence slope allowed to differ between the highest- and lowest-BTD "
            "terciles. Hanlon predicts the high-BTD slope is **lower**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.earnings_persistence(EV)\n"
            "    b_low, b_high, diff, tdiff = q['b_low'], q['b_high'], q['diff'], q['t_diff']\n"
            "    fr = EV.dropna(subset=['roa','roa_next','btd_assets'])\n"
            "    fr = fr[(fr['roa'].abs()<1)&(fr['roa_next'].abs()<1)]\n"
            "    b3 = st._bucketize(fr['btd_assets'].to_numpy(), 3)\n"
            "    xr, yr = fr['roa'].to_numpy(), fr['roa_next'].to_numpy()\n"
            "else:\n"
            "    b_low, b_high, diff, tdiff = R['b_low'], R['b_high'], R['pers_diff'], R['pers_t']\n"
            "    b3 = xr = yr = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if b3 is not None:\n"
            "    for bk,col,lab in [(0,GREEN,'low-BTD (clean)'),(2,GREY,'high-BTD (aggressive)')]:\n"
            "        m=(b3==bk)&(np.abs(xr)<.5)&(np.abs(yr)<.5)\n"
            "        a1.scatter(xr[m]*100, yr[m]*100, s=9, alpha=.30, color=col, label=lab)\n"
            "    a1.set_xlabel('this-year pretax ROA (%)'); a1.set_ylabel('NEXT-year ROA (%)')\n"
            "    a1.set_title('Same slope in both groups'); a1.legend(fontsize=8)\n"
            "else:\n"
            "    a1.text(.5,.5,'run with cache',ha='center'); a1.set_axis_off()\n"
            "a2.bar(['low-BTD\\n(clean)','high-BTD\\n(aggressive)'], [b_low, b_high], color=[GREEN, GREY], width=.5)\n"
            "for i,v in enumerate([b_low,b_high]): a2.annotate(f'{v:.3f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylim(0,1.0); a2.set_ylabel('earnings persistence slope')\n"
            "a2.set_title(f'Interaction {diff:+.3f} (t={tdiff:+.2f}) — flat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'persistence: low-BTD {b_low:.3f}  high-BTD {b_high:.3f}  interaction {diff:+.3f} (t={tdiff:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: this is the study's decisive result — the persistence slopes are "
            f"**identical** (low-BTD {R['b_low']:.3f}, high-BTD {R['b_high']:.3f}, interaction "
            f"**{R['pers_diff']:+.3f}**, *t* = {R['pers_t']:+.2f}). On these names a large book-tax "
            "gap does **not** mark fragile earnings. Unlike a signal whose accounting mechanism is "
            "bulletproof even when the return edge is priced away, here **even the mechanism is "
            "absent** — the return null is not 'efficiently priced real information', it's 'no "
            "information on this tape'. (Pooled *t*, so read the near-zero magnitude.)"
        ),
        md(
            "### 4e · Tradability — the timer\n\n"
            "For completeness, the calendar long-short net of one-way costs × turnover (both legs) "
            "+ short borrow — though a sub-2 gross *t* already settles it."
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
            f"> 💡 In plain words: turnover is low (~{R['ls_turn']:.2f}/mo — a slow annual "
            f"balance-sheet signal), so costs only trim ~+{R['ls_ann']-R['net'][(20,100)][1]:.1f}"
            f"%/yr. But that's irrelevant: net NW *t* = {R['net'][(20,100)][2]:+.2f}, and you "
            "cannot be paid for a spread that wasn't distinguishable from zero to begin with. "
            "**Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + signal panel with a TUNABLE planted Hanlon effect (low-BTD names "
            "drift up). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=856 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.35, seed=856)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted effect (edge=0.35)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null never fires; a planted effect lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 "
            f"**{R['syn_null_fire']}/12** times; a planted effect reads NW *t* = "
            f"{R['syn_planted_t']:.2f}. The machinery is unbiased and powered, so the real-tape "
            f"+{R['ls_t_nw']:.2f} is a genuine near-null, not a broken pipeline. *(Power check "
            "only — never cited in support of a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `WEAK`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo (+{R['ls_ann']:.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; change variant NW *t* = "
            f"{R['change_t_nw']:+.2f}; pooled event drift **mildly anti-Hanlon** and insignificant "
            "(placebo *p* > 0.5); both eras right-signed but insignificant. Right sign not even "
            "stable across methods — tape-uncertified.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: +"
            f"{R['net'][(20, 100)][1]:.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20, 100)][3]:.2f}; fails before costs.\n"
            f"- **Marks fragile earnings? `NOT DETECTED`** — persistence interaction "
            f"{R['pers_diff']:+.3f} (*t* = {R['pers_t']:+.2f}); high-BTD {R['b_high']:.3f} vs "
            f"low-BTD {R['b_low']:.3f}. The mechanism that anchors the claim is flat on blue chips."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Where the effect probably still lives** is the **broad cross-section** Hanlon "
            "studied — small, distressed, tax-shelter-heavy names with genuinely large, "
            "under-scrutinised book-tax gaps. A large-cap *survivor* basket is the hardest possible "
            "test: these are the firms where the gap is smallest, best-explained and most-watched. "
            "The null here refutes the **blue-chip** version, not the anomaly itself.\n"
            "- **Coverage honesty:** the cross-section runs ≈"
            f"{R['xsec_early']:.0f} → {R['xsec_late']:.0f} names, three names lack a clean "
            "pretax-income tag, and the statutory-rate gross-up ignores the 2018 blended-rate year. "
            "A wider, survivorship-free panel with hand-checked tax footnotes is the honest next "
            "step.\n"
            "- **Dedup map:** [568-effective-tax-rate](../../568-effective-tax-rate/) (the tax "
            "*rate* level/change, not the book-tax *gap*), [231-sloan-accruals](../../231-sloan-accruals/) "
            "(total accruals, the family BTD is a tax-specific slice of), "
            "[229-beneish-m-score](../../229-beneish-m-score/) (an eight-signal manipulation score). "
            "None ranks on the statutory-grossed-up book-minus-tax income gap.\n\n"
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
