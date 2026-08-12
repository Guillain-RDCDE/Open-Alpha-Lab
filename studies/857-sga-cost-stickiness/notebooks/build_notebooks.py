"""Generate the two narrative notebooks for Study 857 (SG&A Cost Stickiness).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR SG&A + revenue + net
# income + assets + yfinance prices, 33 names / 25 identified, ends 2015-09 -> 2026-05,
# as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=33, n_ident=25, n_events=314,
    end_lo="2015-09-26", end_hi="2026-05-03", fp_prices="2cf202cb4854",
    # ABJ aggregate replication (pooled firm-quarter estimates)
    abj_b1=0.651, abj_b1p2=0.554, abj_b2=-0.097,
    n_sticky=12, n_total=25,
    # primary calendar long-short (disc = -stickiness, terciles, staleness 200)
    n_months=101, avg_n=11.0, ls_span_lo="2015-12-31", ls_span_hi="2026-06-30",
    ls_mean_bps=-28.6, ls_ann=-3.43, ls_t_iid=-0.57, ls_t_nw=-0.45, ls_sharpe=-0.20,
    ls_hit=48, ls_long_bps=125.2, ls_short_bps=153.7, ls_turn=0.07, ls_cum=0.66,
    ls120_mean_bps=28.1, ls120_t_nw=0.66,
    # pooled event drift  horizon -> (n, top%, bot%, ls%, t, win%, placebo p)
    drift={
        21: (313, 1.37, 1.72, -0.35, -0.32, 50, 0.6233),
        63: (301, 5.92, 4.65, 1.27, 0.57, 54, 0.2546),
        126: (296, 10.77, 11.52, -0.76, -0.24, 54, 0.5872),
    },
    mono63=(4.65, 4.85, 5.92),
    # era split (calendar LS), split at 2020-07-01
    era_early_n=54, era_early_bps=14.4, era_early_t=0.20,
    era_late_n=47, era_late_bps=-78.0, era_late_t=-0.74,
    # third axis — stickiness -> future ROA change
    roa_n=226, roa_slope=0.0004, roa_t=0.15, roa_r2=0.000, roa_corr=0.010,
    roa_top=0.34, roa_bot=0.48, roa_spread=-0.14,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (-34.1, -4.10, -0.54, -0.23), (20, 100): (-39.7, -4.77, -0.63, -0.27)},
    # synthetic control
    syn_null_mean=0.02, syn_null_sd=0.87, syn_null_fire=0, syn_null_seeds=10,
    syn_planted_bps=156.9, syn_planted_t=3.16,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Sticky costs%3F: Confirmed](https://img.shields.io/badge/Sticky_costs%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from sga_stickiness import data, strategy as st

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
            "# Costs are 'sticky'. Can you trade the firms that don't cut them? 📎\n"
            "### A real, textbook accounting fact — SG&A clings on the way down — that turns out "
            "to sort *nothing* in the stock market\n\n"
            + BADGES +
            "Here's an accounting fact with a name and a famous paper behind it. When sales rise, "
            "a company's overhead — salespeople, marketing, admin, the SG&A line — rises with it. "
            "But when sales *fall*, that overhead comes down **more slowly**: managers hesitate to "
            "fire the sales team or kill a campaign they might need again next quarter. Costs are "
            "**sticky** on the way down. Anderson, Banker & Janakiraman measured it in 2003: SG&A "
            "rises ~0.55% per 1% sales gain but falls only ~0.35% per 1% sales loss.\n\n"
            "The trading pitch writes itself: a firm whose costs stay sticky into a downturn is "
            "**badly disciplined** — it should earn less. So short the sticky firms, buy the lean "
            "ones. Sounds airtight. It isn't.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 33 large US filers that report SG&A on EDGAR (only 25 ever "
            "*identify* stickiness — you need real sales declines to measure it), 2015→2026; a "
            "genuinely **thin, cyclical-tilted, noisy panel**. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Are SG&A costs really **sticky**? | **Yes.** Across the panel SG&A rises "
            f"**+{R['abj_b1']:.2f}%** per 1% sales gain but falls only **+{R['abj_b1p2']:.2f}%** "
            f"per 1% sales loss (β₂ = {R['abj_b2']:+.2f}). The 2003 effect replicates. |\n"
            f"| Do **sticky-cost firms under-earn**? | **No — not in any direction we can find.** "
            f"A long-short that buys the leanest firms and shorts the stickiest earns "
            f"**{R['ls_ann']:+.1f}%/yr** — the *wrong sign* and indistinguishable from luck "
            f"(robust *t* = **{R['ls_t_nw']:+.2f}**), and it flips sign if you nudge the "
            "settings. |\n"
            f"| Does stickiness at least predict weaker **profits**? | **No.** The stickiest third "
            f"of firms go on to earn essentially the same as the leanest — a ROA-change gap of "
            f"**{R['roa_spread']:+.2f} pp** (correlation +{R['roa_corr']:.2f}). Nothing. |\n"
            "| Can you trade it? | **No.** It's the wrong sign and insignificant before costs — "
            "there's nothing to charge costs against. |\n\n"
            "> A real, measurable behaviour of costs. No signal in the stock, no signal in the "
            "future earnings. That gap is the whole study."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"SG&A is sticky — it falls slower than it rises. A firm that lets costs cling "
            "into a sales decline is undisciplined and should under-earn. Short the sticky, buy "
            "the lean.\"*\n\n"
            "The premise (costs are sticky) is **true and well-replicated** — it's one of the "
            "most cited results in management accounting. The leap is the second sentence: that a "
            "regression coefficient anyone can compute from years-old public filings tells you "
            "which stocks will win."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a firm's *cost-cutting reflexes* — estimable from its own 10-Ks — predicted its "
            "stock, that would be a tidy quality anomaly: no alt-data, just the income statement. "
            "That's exactly why it deserves suspicion. Everyone can run the ABJ regression. If "
            "markets price public operating characteristics at all, a decade-old cost elasticity "
            "is the kind they should have long since absorbed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** For each firm, estimate its **cost stickiness** (how much less "
            "SG&A falls on the way down than it rises on the way up) using *only* the filings "
            "public at each point in time — no peeking at the future.\n"
            "- **The return test.** Each month, rank firms from leanest to stickiest, buy the "
            "lean third, short the sticky third, hold a month. Does the spread make money — and "
            "can we tell it from noise?\n"
            "- **The profits test.** Do the stickiest firms actually go on to earn less (weaker "
            "return on assets)? If the 'weak discipline' story is right, they should.\n"
            "- **The mirage check.** If the return spread can't beat a coin-flip relabelling of "
            "the firms, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the part that's true: are costs really sticky?** For every firm-quarter we "
            "line up the year-over-year change in SG&A against the change in sales, separately for "
            "quarters when sales rose and quarters when they fell."
        ),
        code(
            "up, dn = R['abj_b1'], R['abj_b1p2']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['sales RISE\\n(per +1%)', 'sales FALL\\n(per -1%)'], [up, dn],\n"
            "       color=[GREY, RED], width=.55)\n"
            "for i,v in enumerate([up, dn]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('SG&A response (% move)')\n"
            "ax.set_title(f'Costs are sticky: SG&A falls {up-dn:+.2f}% LESS on the way down (ABJ replicates)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SG&A rises {up:+.2f}% per +1% sales, falls only {dn:+.2f}% per -1% sales '\n"
            "      f'-> asymmetry β₂ = {R[\"abj_b2\"]:+.3f} (sticky).')"
        ),
        md(
            f"There it is — SG&A comes down **{R['abj_b1'] - R['abj_b1p2']:+.2f} percentage "
            f"points less** per unit of sales decline than it goes up per unit of sales gain. The "
            "Anderson-Banker-Janakiraman effect is real and it replicates on our names. Costs "
            "genuinely cling on the way down.\n\n"
            "**So now the money question: do the sticky firms under-earn?** Same firms, ranked "
            "from leanest to stickiest; we buy the lean third and short the sticky third and "
            "measure the forward *return*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='disc', n_buckets=3, min_names=6, staleness_days=200)\n"
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
            "    ax.set_ylabel('growth of $1 (gross, long lean / short sticky)')\n"
            "    ax.set_title(f'The discipline bet LOSES money: {ann:+.1f}%/yr, robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long lean / short sticky: {ann:+.1f}%/yr gross, Newey-West t = {tnw:+.2f} (the bar is |t|>=2)')"
        ),
        md(
            f"It **slopes down**. Buying discipline and shorting stickiness lost money — "
            f"**{R['ls_ann']:+.1f}%/yr**, shrinking \\$1 to \\${R['ls_cum']:.2f} over "
            f"{R['n_months']} months. That's not just 'no edge', it's the *opposite* of the "
            f"claim's sign — the sticky firms slightly out-returned the lean ones. But before you "
            f"flip the trade: the robust *t* is **{R['ls_t_nw']:+.2f}**, nowhere near "
            f"significance, and it **flips to positive** if you change how stale a signal you'll "
            "tolerate. It's noise wearing a sign.\n\n"
            "**The tell:** if stickiness really sorted returns, the middle third would sit between "
            "the ends. And when we look across horizons, the spread can't even hold one sign."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for h in st.HORIZONS:\n"
            "        es = st.event_summary(PX, EV, horizon=h, n_buckets=3, n_draws=3000)\n"
            "        rows.append((h, es['ls_mean']*100, es['t'], es['p_placebo']))\n"
            "else:\n"
            "    for h in st.HORIZONS:\n"
            "        d = R['drift'][h]; rows.append((h, d[3], d[4], d[6]))\n"
            "hs=[f'{r[0]}d' for r in rows]; vals=[r[1] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(hs, vals, color=[RED if v<0 else GREY for v in vals], width=.55)\n"
            "for i,(h,v,t,p) in enumerate(rows): ax.annotate(f'{v:+.2f}%\\n(p={p:.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('lean-minus-sticky forward return')\n"
            "ax.set_title('No stable sign: the spread wanders around zero across horizons')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('lean-minus-sticky drift by horizon:', [f'{v:+.2f}% (placebo p={p:.2f})' for h,v,t,p in rows])"
        ),
        md(
            "The spread is negative at 1 month, positive at 1 quarter, negative again at 2 "
            "quarters — and a **random relabelling of the firms beats the real ranking** a quarter "
            "to six-tenths of the time (placebo *p* ≈ 0.25-0.62). There is no ladder and no stable "
            "direction. Whatever cost stickiness is telling you, it isn't telling you which stock "
            "to own."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Is SG&A sticky? — Confirmed.** SG&A rises +{R['abj_b1']:.2f}% per 1% sales gain, "
            f"falls only +{R['abj_b1p2']:.2f}% per 1% sales loss. The ABJ phenomenon is real.\n"
            f"- **Signal (returns) — None.** The discipline long-short is the wrong sign "
            f"({R['ls_ann']:+.1f}%/yr), insignificant (robust *t* = {R['ls_t_nw']:+.2f}), "
            "sign-unstable, and the event drift can't hold a direction. No effect either way.\n"
            f"- **Weaker future profits? — No.** The stickiest third's next-year ROA change is "
            f"only {R['roa_spread']:+.2f} pp different from the leanest — statistically nothing.\n"
            "- **Tradability — Mirage.** You can't get paid for a spread that isn't there.\n\n"
            "> The honest one-liner: *costs really are sticky — and the market could not care "
            "less which firms let them stick.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where an edge might still hide.** Not in the *level* of a public cost elasticity, "
            "but maybe in the *surprise* — a firm that suddenly stops cutting when analysts expect "
            "discipline — or in stickiness interacted with leverage or a live downturn. That's the "
            "residual this study didn't chase.\n"
            "- **The coverage caveat is real.** Only firms that actually *decline* identify "
            f"stickiness, so the panel is thin ({R['n_ident']} of {R['n_names']} names), "
            "cyclical-tilted, and the quarterly firm-level β₂ is noisy — attenuation could bury a "
            "weak true effect. But the profits regression is a flat zero too, which makes a hidden "
            "return edge unlikely.\n"
            "- **Sibling studies:** [524-operating-leverage](../../524-operating-leverage/) (the "
            "*size* of the cost-to-sales elasticity), [200-roe-quality](../../200-roe-quality/) "
            "and [122-gross-profitability](../../122-gross-profitability/) (profitability "
            "*levels*), [749-layoff-drift](../../749-layoff-drift/) (an announced cost-cut "
            "*event*). See [docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the alpha is in the stickiness *surprise*, not the level? Build the "
            "expectation model, show a certifiable net spread on the size you'd actually run — "
            "then we'll talk.*"
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
            "# SG&A Cost Stickiness — a quantitative teardown 🔬\n"
            "### A point-in-time ABJ estimator · a calendar-time tercile long-short (Newey-West) · "
            "a pooled event-drift cross-check with a label-shuffle placebo · an era split · a "
            "future-ROA regression · a 10-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **sticky-cost firms under-earn** — rests on a real phenomenon (Anderson-"
            "Banker-Janakiraman cost stickiness) and a leap (that a firm's cost-cut elasticity "
            "sorts its stock). We confirm the phenomenon and reject the leap. This is distinct "
            "from every sibling: [524](../../524-operating-leverage/) is the cost elasticity's "
            "*magnitude*, [200](../../200-roe-quality/) / [122](../../122-gross-profitability/) "
            "are profitability *levels*, [749](../../749-layoff-drift/) is a cost-cut *event*.\n\n"
            "> ⚠️ **Data note.** EDGAR `SellingGeneralAndAdministrativeExpense` + revenue "
            "(quarterly) + net income + assets, " + str(R["n_names"]) + " names ("
            + str(R["n_ident"]) + " identify β₂), ends " + R["end_lo"] + " → " + R["end_hi"]
            + ", as-of " + R["as_of"] + ". Stickiness estimated **point-in-time on an expanding "
            "window** of only-public filings. Survivorship named on the Signal axis (current "
            "survivors — the bias runs *against* the claim). Thin/cyclical/noisy coverage is a "
            "first-class caveat. Numbers in [`docs/results.md`](../docs/results.md) (prices "
            "fingerprint `" + R["fp_prices"] + "`).\n"
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
            f"*t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}** — wrong sign, "
            f"sign-unstable (staleness-120 → {R['ls120_t_nw']:+.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: NW *t* = "
            f"{R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; wrong sign "
            "before costs |\n"
            f"| **Is SG&A sticky?** | `CONFIRMED` | pooled β₁ = +{R['abj_b1']:.2f} vs β₁+β₂ = "
            f"+{R['abj_b1p2']:.2f} (β₂ = {R['abj_b2']:+.2f}); future-ROA slope {R['roa_slope']:+.4f} "
            f"(corr +{R['roa_corr']:.2f}), tercile spread {R['roa_spread']:+.2f} pp |\n\n"
            "> 💡 In plain words: the accounting regularity is genuine and the return/profit edge "
            "is a two-sided null. A public cost elasticity describes how costs behave, and the "
            "market neither rewards nor punishes the firms that let them stick."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For firm $i$ at quarter $q$, regress the point-in-time year-over-year log changes on "
            "an expanding public window,\n\n"
            "$$\\Delta\\log SGA_{i,q} = \\beta_0 + \\beta_1\\,\\Delta\\log Sales_{i,q} + "
            "\\beta_2\\,(D_{i,q}\\cdot\\Delta\\log Sales_{i,q}) + \\varepsilon,$$\n\n"
            "with $D=1$ when sales fell YoY. $\\beta_1$ is the up-response, $\\beta_1+\\beta_2$ the "
            "down-response, and **stickiness $= -\\beta_2$**. We trade $disc_{i,q}=\\beta_2$ "
            "(= $-$stickiness; higher ⇒ leaner). The claims:\n\n"
            "- **H₁ (stickiness exists).** $\\beta_2 < 0$ on average — costs are sticky.\n"
            "- **H₂ (sticky firms under-earn).** A long-lean / short-sticky spread on $disc$ is "
            "positive.\n"
            "- **H₃ (weaker operating results).** Higher stickiness ⇒ weaker future ROA.\n"
            "- **H₄ (tradable).** Any H₂ spread survives costs + borrow.\n\n"
            "We find **H₁ confirmed** (pooled β₁ = +"
            f"{R['abj_b1']:.2f} vs β₁+β₂ = +{R['abj_b1p2']:.2f}), **H₂ rejected** (NW *t* = "
            f"{R['ls_t_nw']:+.2f}, wrong sign, sign-unstable), **H₃ rejected** (ROA slope "
            f"{R['roa_slope']:+.4f}, corr +{R['roa_corr']:.2f}), and therefore **H₄ moot**. Note "
            "H₂ is `NONE`, not `WEAK`: there is no consistent right-signed whiff to certify — the "
            "point estimates straddle zero."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, because a "
            "firm-level characteristic is persistent and filings cluster: a monthly long-short "
            "series lets a **Newey-West (6-lag) HAC *t*** do the honest work the desk's `REAL` bar "
            "is written against. The panel is thin, so we sort into **terciles** (not quintiles) "
            "and require ≥ 6 names. The pooled event drift + a **label-shuffle placebo** is the "
            "cross-check; the future-ROA regression is graded on **magnitude** (its pooled *t* "
            "ignores firm/quarter clustering). Crucially, β₂ is estimated **point-in-time** — the "
            "expanding window at each filing uses only prior public data, so the signal never "
            "peeks at its own future."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) stickiness estimates across "
            f"{R['n_ident']} names that identify β₂ (of {R['n_names']} in the basket), ends "
            f"{R['end_lo']} → {R['end_hi']}, each stamped with its 10-Q/10-K filing date.\n"
            "- **Estimator.** ABJ asymmetric regression on an expanding window of public quarterly "
            "YoY changes; guards: ≥20 obs, ≥4 decline quarters (β₂ must be identified).\n"
            "- **Primary.** Monthly tercile long-short on `disc`, one execution lag; Newey-West + "
            "one-sample *t*, Sharpe, hit rate.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag entry, "
            "one-sample *t* + label-shuffle placebo, tercile monotonicity.\n"
            "- **Robustness.** Staleness 120 vs 200 days; an era split at the 2020 sales shock.\n"
            "- **Mechanism.** Pooled OLS of next-year ROA change on stickiness.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short "
            "borrow.\n"
            "- **Control.** Synthetic SG&A/sales levels through the *same* estimator; the null "
            "must not fire across 10 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The phenomenon — is SG&A actually sticky? (ABJ replication)\n\n"
            "Pooled across all firm-quarter estimates, the average up-response vs down-response."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b1, b2 = EV['beta1'].mean(), EV['beta2'].mean()\n"
            "else:\n"
            "    b1, b2 = R['abj_b1'], R['abj_b2']\n"
            "up, dn = b1, b1 + b2\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar(['β₁  (sales +1%)', 'β₁+β₂  (sales -1%)'], [up, dn], color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([up, dn]): ax.annotate(f'{v:+.3f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('SG&A elasticity'); ax.axhline(0,c='k',lw=.6)\n"
            "ax.set_title(f'Sticky: SG&A falls {up-dn:+.3f} LESS per unit decline (β₂={b2:+.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pooled ABJ: β₁={b1:+.3f} (up), β₁+β₂={b1+b2:+.3f} (down), asymmetry β₂={b2:+.3f}')\n"
            "print(f'{R[\"n_sticky\"]} of {R[\"n_total\"]} names estimate as sticky (final β₂<0)')"
        ),
        md(
            f"> 💡 In plain words: β₁ = +{R['abj_b1']:.2f} but β₁+β₂ = +{R['abj_b1p2']:.2f} — SG&A "
            f"comes down **less** than it goes up. The ABJ effect is real on our tape (milder than "
            "the original 0.55/0.35 because we estimate at quarterly YoY frequency). The input "
            "phenomenon is not in doubt. Everything below asks whether it *sorts stocks*."
        ),
        md(
            "### 4b · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh `disc` into terciles each month, long leanest / short stickiest, earn next "
            "month's return. The decisive statistic is the HAC *t* of the monthly series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='disc', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls120 = st.calendar_ls(PX, EV, signal_col='disc', n_buckets=3, min_names=6, staleness_days=120)\n"
            "    s120 = st.calendar_ls_stats(ls120)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo ({s['ann_pct']:+.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  staleness-120 variant: {s120['mean_bps']:+.1f} bps/mo, NW t = {s120['t_nw']:+.2f}  (sign flips!)\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=RED, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: {R['ls_ann']:+.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & cyclical: only decliners identify β₂')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-120 NW t = {R['ls120_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **{R['ls_mean_bps']:+.0f} bps/month** with a HAC *t* of "
            f"**{R['ls_t_nw']:+.2f}** — the wrong sign for the claim and statistically zero. The "
            f"single most damning detail: nudge the staleness window from 200 to 120 days and the "
            f"mean flips to **{R['ls120_mean_bps']:+.0f} bps** (*t* = {R['ls120_t_nw']:+.2f}). A "
            f"real signal doesn't change sign when you jiggle a housekeeping parameter. n = "
            f"{R['n_months']} months, avg cross-section {R['avg_n']:.0f} — thin by construction."
        ),
        md(
            "### 4c · The cross-check — pooled event drift + placebo + monotonicity\n\n"
            "Bucket all events by `disc`; top-minus-bottom forward drift with a label-shuffle "
            "null. If there were a sort, the terciles would form a ladder and the sign would hold."
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
            "a2.bar(['bottom\\n(sticky)','middle','top\\n(lean)'], mono, color=GREY, width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('Barely a ladder, tiny gap')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift **can't hold a sign** "
            f"({R['drift'][21][3]:+.2f}% / {R['drift'][63][3]:+.2f}% / {R['drift'][126][3]:+.2f}% "
            f"across 1m/1q/2q), the *t* never exceeds +{R['drift'][63][4]:.2f}, and the "
            f"label-shuffle placebo *p* runs **0.25-0.62** — a random tercile split routinely "
            f"beats the real one. The 63-day terciles ({R['mono63'][0]:+.2f}% / "
            f"{R['mono63'][1]:+.2f}% / {R['mono63'][2]:+.2f}%) lean the claim's way by a hair, but "
            "it's the only horizon that does. The event study and the calendar long-short agree: "
            "no return sort."
        ),
        md(
            "### 4d · Era split — nothing hiding in a regime (and the sign flips)\n\n"
            "Split the calendar long-short at the 2020 sales shock."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = ls[ls.index < data.ERA_SPLIT]['ls'].to_numpy(); l = ls[ls.index >= data.ERA_SPLIT]['ls'].to_numpy()\n"
            "    eb, et, en = e.mean()*1e4, st.newey_west_t(e), len(e)\n"
            "    lb, lt, ln = l.mean()*1e4, st.newey_west_t(l), len(l)\n"
            "else:\n"
            "    eb, et, en = R['era_early_bps'], R['era_early_t'], R['era_early_n']\n"
            "    lb, lt, ln = R['era_late_bps'], R['era_late_t'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar([f'pre-2020H2\\n(n={en})', f'2020H2-2026\\n(n={ln})'], [eb, lb],\n"
            "       color=[GREY, RED], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Opposite-signed, both insignificant — the fingerprint of noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2020H2: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2020H2-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: {R['era_early_bps']:+.0f} bps (NW *t* = {R['era_early_t']:+.2f}) "
            f"early, {R['era_late_bps']:+.0f} bps (NW *t* = {R['era_late_t']:+.2f}) late — "
            "**opposite signs**, both insignificant. Not a decayed edge; a coin that landed "
            "differently in each half."
        ),
        md(
            "### 4e · The mechanism check — does stickiness predict weaker profits?\n\n"
            "Pooled OLS of the next-year change in trailing ROA on the stickiness measure, and the "
            "future-ROA-change spread between the stickiest and leanest terciles."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.predicts_profitability(EV)\n"
            "    fr = EV.dropna(subset=['stickiness','roa_chg'])\n"
            "    x, y = fr['stickiness'].to_numpy(), fr['roa_chg'].to_numpy()\n"
            "    slope, corr, top, bot = q['slope'], q['corr'], q['top_roa_chg']*100, q['bot_roa_chg']*100\n"
            "else:\n"
            "    x = y = None\n"
            "    slope, corr, top, bot = R['roa_slope'], R['roa_corr'], R['roa_top'], R['roa_bot']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if x is not None:\n"
            "    m = (np.abs(x)<2)&(np.abs(y)<0.2)\n"
            "    a1.scatter(x[m], y[m]*100, s=9, alpha=.3, color=GREY)\n"
            "    xs = np.linspace(np.percentile(x,2), np.percentile(x,98), 50)\n"
            "    a1.plot(xs, (q['slope']*xs + (y.mean()-q['slope']*x.mean()))*100, color=RED, lw=2)\n"
            "    a1.set_xlabel('stickiness (−β₂)'); a1.set_ylabel('next-year ROA change (pp)')\n"
            "    a1.set_title(f'No relation: slope {slope:+.4f}, corr {corr:+.2f}')\n"
            "else:\n"
            "    a1.text(.5,.5,'run with cache',ha='center'); a1.set_axis_off()\n"
            "a2.bar(['lean third','sticky third'], [bot, top], color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([bot, top]): a2.annotate(f'{v:+.2f}pp',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('next-year ROA change'); a2.set_title(f'Trivial gap: {top-bot:+.2f} pp')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'stickiness -> future ROA: slope {slope:+.4f}, corr {corr:+.3f}, tercile spread {top-bot:+.2f} pp (sticky {top:+.2f} vs lean {bot:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the 'weak discipline → weak earnings' mechanism **doesn't show "
            f"up** either. The slope is {R['roa_slope']:+.4f}, the correlation +{R['roa_corr']:.2f}, "
            f"and the stickiest third's next-year ROA change is only {R['roa_spread']:+.2f} pp "
            "different from the leanest — directionally the claim's way, magnitudinally nothing. "
            "(A noisy quarterly β₂ attenuates any true relation; but a flat return null *and* a "
            "flat profit null is a consistent 'nothing here'.)"
        ),
        md(
            "### 4f · Tradability — the timer\n\n"
            "For completeness, the calendar long-short net of one-way costs × turnover (both "
            "legs) + short borrow — though a wrong-signed sub-1 gross *t* already settles it."
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
            "ax.set_title('Wrong sign gross; costs only deepen the loss')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: {a:+.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: net of 20 bps + 100 bps borrow the spread is "
            f"{R['net'][(20,100)][1]:+.2f}%/yr, NW *t* = {R['net'][(20,100)][2]:+.2f}. There is no "
            "spread to pay costs on — it was the wrong sign and insignificant to begin with. "
            "**Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Faithful-engine & power control\n\n"
            "Synthetic SG&A + sales *levels* obeying a per-firm true β₂, pushed through the "
            "**same** estimator, with a TUNABLE planted forward-return link (lean names drift up). "
            "The null (edge = 0) is checked over **10 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(10):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=857 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.35, seed=857)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,10), null_ts, color=GREY, s=45, label='null worlds (edge=0), 10 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted link (edge=0.35)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 10','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null does not fire; a planted link lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/10  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 10 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 "
            f"**{R['syn_null_fire']}/10** times; a planted link reads NW *t* = {R['syn_planted_t']:.2f}. "
            f"The machinery — including the point-in-time ABJ estimator — is unbiased and powered, "
            f"so the real-tape {R['ls_t_nw']:+.2f} is a genuine null, not a broken pipeline. "
            "*(Power check only — never cited in support of a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `NONE`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo ({R['ls_ann']:+.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}** (wrong sign), sign-unstable "
            f"(staleness-120 → {R['ls120_t_nw']:+.2f}); event drift flat and sign-inconsistent "
            f"(placebo *p* ≈ 0.25-0.62); eras opposite-signed. No robust effect either way.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: "
            f"{R['net'][(20, 100)][1]:+.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20, 100)][3]:.2f}; wrong sign before costs.\n"
            f"- **Is SG&A sticky? `CONFIRMED`** — pooled β₁ = +{R['abj_b1']:.2f} vs β₁+β₂ = "
            f"+{R['abj_b1p2']:.2f} (β₂ = {R['abj_b2']:+.2f}). The ABJ phenomenon is real; it just "
            f"neither sorts returns nor forecasts weaker ROA (spread {R['roa_spread']:+.2f} pp, "
            f"corr +{R['roa_corr']:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The residual worth chasing** is the *surprise*, not the *level*: a firm that "
            "suddenly stops cutting when the market expects discipline, or stickiness interacted "
            "with leverage / a live downturn. A decade-old public cost elasticity that forecasts "
            "neither the stock nor next year's ROA is the textbook signature of an already-priced "
            "(or simply non-predictive) fundamental.\n"
            "- **Coverage honesty:** only firms that *decline* identify β₂, so the panel is thin "
            f"({R['n_ident']} of {R['n_names']} names), cyclical-tilted, and the quarterly "
            "firm-level β₂ is noisy — attenuation is a real caveat, but a flat return null *and* a "
            "flat profit null point the same way.\n"
            "- **Dedup map:** [524-operating-leverage](../../524-operating-leverage/) (cost "
            "elasticity *magnitude*), [200-roe-quality](../../200-roe-quality/) & "
            "[122-gross-profitability](../../122-gross-profitability/) (profitability *levels*), "
            "[749-layoff-drift](../../749-layoff-drift/) (an announced cost-cut *event*). None "
            "estimates the asymmetric SG&A response itself.\n\n"
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
