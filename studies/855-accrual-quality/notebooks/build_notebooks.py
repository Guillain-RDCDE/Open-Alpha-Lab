"""Generate the two narrative notebooks for Study 855 (Accrual Quality, Dechow-Dichev).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR NI/CFO/assets + yfinance
# prices, 42 deep-history non-financial names, period ends 2010-10 -> 2026-03, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=42, n_events=2066,
    end_lo="2010-10-31", end_hi="2026-03-31", fp_prices="f0610f2cc06e",
    # primary calendar long-short (quality=-aq_vol, terciles, staleness 200)
    n_months=182, avg_n=33.5, ls_span_lo="2011-05-31", ls_span_hi="2026-06-30",
    ls_mean_bps=-3.6, ls_ann=-0.44, ls_t_iid=-0.19, ls_t_nw=-0.18, ls_sharpe=-0.05,
    ls_hit=49, ls_long_bps=124.3, ls_short_bps=127.9, ls_turn=0.07, ls_cum=0.88,
    ls120_mean_bps=-6.7, ls120_t_nw=-0.33,
    wc_mean_bps=-10.7, wc_t_nw=-0.47, wc_sharpe=-0.11,
    xsec_early=25.9, xsec_late=39.7,
    # pooled event drift  horizon -> (n, top%, bot%, ls%, t, win%, placebo p)
    drift={
        21: (2066, 1.01, 1.25, -0.24, -0.58, 51, 0.7269),
        63: (2038, 3.40, 3.96, -0.56, -0.79, 52, 0.7985),
        126: (1999, 7.19, 8.17, -0.98, -0.91, 53, 0.8444),
    },
    mono63=(3.96, 3.36, 3.40),
    # era split (calendar LS)
    era_early_n=80, era_early_bps=0.4, era_early_t=0.02,
    era_late_n=102, era_late_bps=-6.8, era_late_t=-0.23,
    # persistence mechanism
    p_n=2066, good_slope=0.870, good_t=19.97, poor_slope=0.273, poor_t=8.59,
    slope_gap=0.596, good_earn_vol=0.53, poor_earn_vol=2.44,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (-9.2, -1.10, -0.46, -0.12), (20, 100): (-14.7, -1.77, -0.73, -0.20)},
    # synthetic control
    syn_null_mean=0.10, syn_null_sd=1.08, syn_null_fire=1, syn_null_seeds=12,
    syn_planted_bps=982.0, syn_planted_t=23.79,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Earnings persistence%3F: Confirmed](https://img.shields.io/badge/Earnings_persistence%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from accrual_quality import data, strategy as st

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
            "# \"Accrual quality\" sorts good earnings from noisy ones. Can you trade it? 📊\n"
            "### A textbook accounting-quality measure that *genuinely* flags flaky earnings — "
            "and that the stock market shrugs at\n\n"
            + BADGES +
            "Here's a signal with real accounting logic. Reported earnings are cash flow plus "
            "**accruals** — the bookkeeping that shifts timing (a credit sale is revenue now, "
            "cash later). Dechow & Dichev (2002) noticed that some firms' accruals map cleanly "
            "into the cash that follows, while others' are mostly guesswork that reverses later. "
            "Measure how *badly* a firm's accruals track its cash flow — the wobble left over "
            "after you line them up — and you've got **accrual quality**. High wobble = "
            "low-quality, noisy earnings. The pitch: low-quality earnings are discounted, so buy "
            "the clean-quality names and short the noisy ones.\n\n"
            "Half of that is dead right. The other half — the part with the money in it — isn't.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 45 deep-history US non-financial names (42 clear the data bar) "
            "with clean EDGAR fundamentals, 2010→2026; a genuinely **thin, uneven panel** (the "
            "rolling window needs ~4 years before a first signal). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does accrual quality flag real **earnings** differences? | **Yes — strongly.** "
            f"The clean-quality names have earnings that **persist** (this quarter's "
            f"profitability carries +0.87 into next quarter) and are ~**5× less volatile** than "
            f"the noisy-quality names (+0.27 carry, {R['poor_earn_vol']:.1f}% vs "
            f"{R['good_earn_vol']:.1f}% swings). The measure works exactly as advertised. |\n"
            f"| Does it predict future **returns**? | **No.** A long-short that buys "
            f"high-quality names and shorts low-quality ones earns **{R['ls_ann']:+.1f}%/yr** — "
            f"i.e. essentially zero, and if anything a hair the *wrong* way (robust *t* = "
            f"**{R['ls_t_nw']:+.2f}**, the bar is 2). |\n"
            "| Why the gap? | \"This firm has noisier earnings\" is a slow, **public** "
            "fundamental. Anyone can compute it from the filings; the market has long since "
            "priced whatever discount noisy earnings deserve. A real fact about the *business* "
            "is not automatically a signal about the *stock*. |\n"
            "| Can you trade it? | **No.** The spread doesn't clear the bar even before costs — "
            "there's nothing to charge costs against. |\n\n"
            "> Real information about earnings quality. No alpha in the stock. That distinction "
            "is the whole study."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Earnings whose accruals don't line up with cash flows are low-quality — noisier, "
            "less reliable, and discounted. So buy the names whose accruals track cash cleanly, "
            "and short the ones whose accruals are all estimation noise.\"*\n\n"
            "It's a specific case of a respectable academic idea — the market misprices the "
            "*accrual* pieces of earnings (Sloan 1996). Dechow-Dichev sharpen it from *how much* "
            "accrual there is to *how reliable* it is: regress accruals on the cash flows they're "
            "supposed to anticipate, and the **leftover wobble** is your quality score."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a number you can compute from three lines of every cash-flow statement predicted "
            "stock returns, it would be one of the easiest anomalies going — no estimates feed, "
            "no alt-data. That's exactly why it deserves suspicion: it's *too* readable. Everyone "
            "with the filings can measure accrual quality. If markets price public information at "
            "all, this is the kind they should price fastest."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** For each of {R['n_names']} deep-history names, the **wobble** "
            "(residual standard deviation) left after regressing accruals on lagged / current / "
            "next-quarter operating cash flow, over a rolling 12-quarter window — known only on "
            "the **filing date** (never using a future cash-flow number that isn't public yet).\n"
            "- **The earnings test.** Do the clean-quality names actually have steadier, more "
            "persistent earnings? (The accounting claim.)\n"
            "- **The return test.** Each month, rank the names on quality, buy the top third, "
            "short the bottom third, hold for the next month. Does the spread make money — and "
            "can we tell it apart from noise?\n"
            "- **The mirage check.** If the return spread can't beat a coin-flip relabelling of "
            "the names, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the part that works: does accrual quality really separate steady earners "
            "from noisy ones?** We split the names into best- and worst-quality thirds and ask "
            "how much of this quarter's profitability carries into next quarter (persistence), "
            "and how volatile their earnings are."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.persistence_by_quality(EV)\n"
            "    gs, ps = q['good_slope'], q['poor_slope']\n"
            "    gev, pev = q['good_earn_vol']*100, q['poor_earn_vol']*100\n"
            "else:\n"
            "    gs, ps = R['good_slope'], R['poor_slope']\n"
            "    gev, pev = R['good_earn_vol'], R['poor_earn_vol']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.bar(['low quality\\n(noisy accruals)','high quality\\n(clean accruals)'], [ps, gs],\n"
            "       color=[GREY, GREEN], width=.55)\n"
            "for i,v in enumerate([ps, gs]): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('earnings persistence (next-q ROA on ROA)'); a1.set_title('Clean-quality earnings persist')\n"
            "a2.bar(['low quality','high quality'], [pev, gev], color=[GREY, GREEN], width=.55)\n"
            "for i,v in enumerate([pev, gev]): a2.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('earnings volatility (% of assets)'); a2.set_title('...and are far less volatile')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'persistence: high-quality {gs:+.2f} vs low-quality {ps:+.2f}  |  '\n"
            "      f'earnings vol: high-quality {gev:.2f}% vs low-quality {pev:.2f}% of assets')"
        ),
        md(
            f"That's the accounting claim, **confirmed decisively**. High-quality earnings carry "
            f"**+{R['good_slope']:.2f}** quarter-to-quarter versus only **+{R['poor_slope']:.2f}** "
            f"for low-quality names, and they swing about **5× less** "
            f"({R['good_earn_vol']:.1f}% vs {R['poor_earn_vol']:.1f}% of assets). The Dechow-Dichev "
            "measure is not folklore — it really does sort steady earners from flaky ones.\n\n"
            "**So now the money question: do the stocks follow?** Same ranking, but instead of "
            "earnings we measure the forward *return* of a buy-the-high-quality, short-the-"
            "low-quality portfolio."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='quality', n_buckets=3, min_names=6, staleness_days=200)\n"
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
            "    ax.set_title(f'Buying quality went nowhere: {ann:+.1f}%/yr, robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: {ann:+.1f}%/yr gross, Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"Flat — sliding *down* to \\${R['ls_cum']:.2f} on the dollar over {R['n_months']} "
            f"months (**{R['ls_ann']:+.1f}%/yr**). The robust *t*-statistic is "
            f"**{R['ls_t_nw']:+.2f}** — not just short of the **2** we require, but faintly on the "
            f"*wrong side of zero*. Buying quality and shorting junk earned nothing; if anything "
            "the junk quietly out-earned the quality (which, oddly, is what the *risk-premium* "
            "reading of this literature would predict — more on that in the quant notebook).\n\n"
            "**The tell:** if quality really sorted returns, the middle third should land between "
            "the top and bottom. It doesn't — the thirds are tied, and not even in order."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=63)\n"
            "    mono = st.bucket_means(fr, 3)*100\n"
            "else:\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['bottom third\\n(low quality)','middle\\nthird','top third\\n(high quality)'], mono,\n"
            "       color=[GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 3-month forward return')\n"
            "ax.set_title('No ladder: the return thirds are tied and out of order (non-monotone)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('3-month forward return by quality third (low->high):', [f'{v:+.2f}%' for v in mono])"
        ),
        md(
            "No staircase — and the highest-quality third actually sits a touch *below* the "
            "lowest. The clean-quality names don't reliably out-return the noisy ones over the "
            "next quarter. Compare that to the *earnings* chart above, which is a clean, ordered "
            "separation. Same signal, two totally different answers: it grades the **business's "
            "earnings**, not the **stock**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Does it flag earnings quality? — Confirmed.** High-quality earnings persist "
            f"(+{R['good_slope']:.2f} vs +{R['poor_slope']:.2f}) and are ~5× less volatile "
            f"({R['good_earn_vol']:.1f}% vs {R['poor_earn_vol']:.1f}%). Real, mechanical, not in "
            "dispute.\n"
            "- **Signal (returns) — None.** The return spread is a near-zero "
            f"**{R['ls_ann']:+.1f}%/yr**, faintly the *wrong* sign (robust *t* = "
            f"{R['ls_t_nw']:+.2f}), with a flat, out-of-order event drift. There is no evidence "
            "the market pays for accrual quality — here it slightly *penalises* it.\n"
            "- **Tradability — Mirage.** You can't get paid for an edge that isn't there (and "
            "leans the wrong way). It fails before costs are even charged.\n\n"
            "> The honest one-liner: *accrual quality tells you how flaky a firm's earnings are, "
            "and the market already knows.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The direction twist.** The academic *factor* version of this (Francis et al. "
            "2005) argues poor accrual quality should earn a **higher** return — a premium for "
            "bearing information risk — the opposite of the retail \"buy quality\" pitch. Our "
            "faint wrong-signed tilt is consistent with that risk story, but it's far too weak to "
            "certify *either* direction.\n"
            "- **The coverage caveat is real.** The rolling window needs ~4 years of clean "
            f"quarterly cash-flow history before a first signal, and the cross-section grows from "
            f"≈{R['xsec_early']:.0f} names (2012) to ≈{R['xsec_late']:.0f} (2024+). A wider, "
            "deeper panel might sharpen the return test — though the flat event drift makes a "
            "hidden edge unlikely.\n"
            "- **Sibling studies:** [231-sloan-accruals](../../231-sloan-accruals/) (the accrual "
            "*level*), [522-percent-operating-accruals](../../522-percent-operating-accruals/) "
            "(accruals scaled by earnings), [539-cash-flow-volatility](../../539-cash-flow-volatility/) "
            "(raw cash-flow vol), and [52-smoke-screen](../../52-smoke-screen/) (discretionary-"
            "accrual manipulation). See [docs/references.md](docs/references.md) for the exact "
            "dedup.\n\n"
            "*Think the alpha is in the information-risk premium, not the quality discount? Build "
            "the factor, show a certifiable net spread on the size you'd actually run — then we'll "
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
            "# Accrual Quality (Dechow-Dichev) — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a pooled "
            "event-drift cross-check with a label-shuffle placebo · an era split · a "
            "working-capital-accrual variant · the earnings-persistence mechanism · a 12-seed "
            "synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **low accrual quality is discounted, so long high-quality / short "
            "low-quality earns a positive spread** — splits into two testable pieces, and the two "
            "give opposite answers. This is distinct from every sibling on the desk: "
            "[231](../../231-sloan-accruals/) ranks on the accrual *level*, "
            "[522](../../522-percent-operating-accruals/) on accruals scaled by *earnings*, "
            "[539](../../539-cash-flow-volatility/) on raw *cash-flow* vol, "
            "[52](../../52-smoke-screen/) on *discretionary-accrual manipulation*. This is the "
            "**Dechow-Dichev residual volatility** — the reliability of the accrual-to-cash mapping.\n\n"
            "> ⚠️ **Data note.** EDGAR `NetIncomeLoss` / `NetCashProvidedByUsedInOperating"
            "Activities` (quarterly flows reconstructed from the YTD cumulative chain) + `Assets` "
            "+ yfinance adjusted closes, "
            + R["n_names"].__str__() + " names, ends "
            + R["end_lo"] + " → " + R["end_hi"] + ", as-of " + R["as_of"] + ". Point-in-time on "
            "the **filing date** (the DD lead-CFO term never peeks past it). Survivorship named "
            "on the Signal axis (current-survivors basket). Thin/uneven coverage is a first-class "
            "caveat. Numbers in [`docs/results.md`](../docs/results.md) (prices fingerprint `"
            + R["fp_prices"] + "`).\n"
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
            f"*t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; WC-accrual "
            f"variant NW *t* = {R['wc_t_nw']:+.2f} — zero, faintly wrong-signed |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: NW *t* = "
            f"{R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; wrong-signed "
            "**before** costs |\n"
            f"| **Flags earnings quality?** | `CONFIRMED` | persistence slope "
            f"**+{R['good_slope']:.2f}** (high-Q) vs **+{R['poor_slope']:.2f}** (low-Q); earnings "
            f"vol **{R['good_earn_vol']:.1f}%** vs **{R['poor_earn_vol']:.1f}%** of assets |\n\n"
            "> 💡 In plain words: the accounting construct is bulletproof and the return edge is a "
            "flat null (leaning faintly the wrong way). A public earnings-quality number sorts the "
            "business, and the market has already priced the stocks accordingly."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $a_{i,q} = (NI_{i,q} - CFO_{i,q})/\\overline{TA}$ be name $i$'s asset-scaled "
            "accrual at fiscal quarter $q$ and $x_{i,q} = CFO_{i,q}/\\overline{TA}$. The "
            "Dechow-Dichev regression over a rolling window is\n\n"
            "$$a_{i,t} = \\alpha + \\beta_1 x_{i,t-1} + \\beta_2 x_{i,t} + \\beta_3 x_{i,t+1} "
            "+ \\varepsilon_{i,t},$$\n\n"
            "and **accrual quality** is $\\sigma(\\hat\\varepsilon)$ over the window (a *high* "
            "residual vol = *poor* quality). The signal `quality` $= -\\sigma(\\hat\\varepsilon)$, "
            "known at $q$'s filing (the window only uses $t \\le q-1$, so the lead term "
            "$x_{t+1}=x_q$ is public — no look-ahead). The claims:\n\n"
            "- **H₁ (flags earnings quality).** Low quality ⇒ less-persistent, noisier earnings — "
            "the DD validity mechanism.\n"
            "- **H₂ (priced discount).** A cross-sectional long-short on `quality` earns a "
            "positive forward return spread — the retail mispricing claim.\n"
            "- **H₃ (tradable).** That spread survives realistic long-short costs + borrow.\n\n"
            "We find **H₁ strongly supported** (persistence +"
            f"{R['good_slope']:.2f} vs +{R['poor_slope']:.2f}, earnings vol "
            f"{R['good_earn_vol']:.1f}% vs {R['poor_earn_vol']:.1f}%), **H₂ not supported** (NW "
            f"*t* = {R['ls_t_nw']:+.2f}, flat non-monotone drift, *faintly wrong-signed*), and "
            "therefore **H₃ moot**. Note the sign: Francis, LaFond, Olsson & Schipper (2005) "
            "argue poor accrual quality earns a **risk premium** (low quality → *higher* return), "
            "the opposite of the discount claim — and our sliver of tilt is on *that* side. Either "
            "way it is a null, so the honest stamp is `NONE`, not a rescued `WEAK`."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, precisely "
            "because balance-sheet quality signals are persistent and filings cluster: a calendar "
            "series of monthly long-short returns lets a **Newey-West (6-lag) HAC *t*** do the "
            "honest work the desk's `REAL` bar is written against. The panel is thin, so we sort "
            "into **terciles** (not quintiles) and require ≥ 6 names in the cross-section. The "
            "pooled event drift + a **label-shuffle placebo** is the cross-check; the "
            "earnings-persistence axis is graded on **magnitude** (its pooled *t* ignores quarter "
            "clustering, so we read the +0.87-vs-+0.27 slope gap and the 5× vol gap, not the "
            "literal *t*)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) accrual-quality quarters across "
            f"{R['n_names']} names, ends {R['end_lo']} → {R['end_hi']}, each stamped with its "
            "10-Q/10-K filing date (point-in-time; quarterly flows reconstructed from the YTD "
            "cumulative chain).\n"
            "- **Primary.** Monthly tercile long-short on `quality`, one execution lag (rank at "
            "month $M$ close, earn month $M{+}1$); Newey-West + one-sample *t*, Sharpe, hit rate.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag entry, "
            "top-minus-bottom tercile, one-sample *t* + 10k-draw label-shuffle placebo, and the "
            "tercile monotonicity picture.\n"
            "- **Robustness.** Staleness 120 vs 200 days; the working-capital-accrual variant "
            "(residual vol of `(ΔReceivables+ΔInventory)/avg-assets`); an era split at 2018.\n"
            "- **Mechanism.** ROA persistence slope (next-q ROA on ROA) and earnings volatility, "
            "best- vs worst-quality tercile.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short "
            "borrow.\n"
            "- **Control.** Synthetic panel, planted-relation knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh `quality` into terciles each month, long top (high quality) / short "
            "bottom equal-weight, earn next month's return. The decisive statistic is the HAC *t* "
            "of the monthly long-short series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='quality', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ev_wc = EV.copy(); ev_wc['quality_wc'] = -ev_wc['aq_vol_wc']\n"
            "    ls_wc = st.calendar_ls(PX, ev_wc, signal_col='quality_wc', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s_wc = st.calendar_ls_stats(ls_wc)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo ({s['ann_pct']:+.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  WC-accrual variant: {s_wc['mean_bps']:+.1f} bps/mo, NW t = {s_wc['t_nw']:+.2f}, Sharpe {s_wc['sharpe']:.2f}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=RED, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: {R['ls_ann']:+.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven: the panel widens as histories mature')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-120 NW t = {R['ls120_t_nw']:+.2f}, WC-accrual NW t = {R['wc_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **{R['ls_mean_bps']:+.0f} bps/month** is economically nothing "
            f"({R['ls_ann']:+.1f}%/yr gross), and the HAC *t* is **{R['ls_t_nw']:+.2f}** — on the "
            f"wrong side of zero. Every specification agrees: staleness-120 NW *t* = "
            f"{R['ls120_t_nw']:+.2f}, the working-capital-accrual variant NW *t* = "
            f"{R['wc_t_nw']:+.2f}. There is no positive quality premium in these returns — the "
            "book grew \\$1 into \\$" + f"{R['ls_cum']:.2f}" + " over " + f"{R['n_months']}" +
            " months."
        ),
        md(
            "### 4b · The cross-check — pooled event drift + placebo + monotonicity\n\n"
            "Bucket all events by `quality`; top-minus-bottom (high-Q minus low-Q) forward drift "
            "with a label-shuffle null. If there were a sort, the terciles would form a ladder."
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
            "a1.set_ylabel('one-sample t (long-short drift)'); a1.set_title('Flat & negative: no horizon clears |t|=2')\n"
            "a2.bar(['bottom\\n(low Q)','middle','top\\n(high Q)'], mono, color=GREY, width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('No ladder (non-monotone terciles)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift is **negative at every horizon** "
            f"(*t* from {R['drift'][21][4]:+.2f} to {R['drift'][126][4]:+.2f}), the label-shuffle "
            f"placebo *p* sits at **0.73-0.84** (the observed sits in the *left* tail — a random "
            f"tercile split beats buying quality most of the time), and the terciles are "
            f"**non-monotone** ({R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / "
            f"{R['mono63'][2]:+.2f}% low→high at 63d). The event study and the calendar long-short "
            "agree: no positive return sort — if anything a faint negative one."
        ),
        md(
            "### 4c · Era split — nothing hiding in a regime\n\n"
            "Split the calendar long-short at 2018."
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
            "ax.bar([f'pre-2018\\n(n={en})', f'2018-2026\\n(n={ln})'], [eb, lb], color=[GREY, GREY], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Both eras: flat, neither significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2018: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2018-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: {R['era_early_bps']:+.0f} bps (NW *t* = {R['era_early_t']:+.2f}) "
            f"pre-2018, {R['era_late_bps']:+.0f} bps (NW *t* = {R['era_late_t']:+.2f}) after — "
            "dead flat early, faintly negative late. It was never a live edge; there is nothing "
            "to decay."
        ),
        md(
            "### 4d · The mechanism — accrual quality *does* flag earnings quality\n\n"
            "The DD validity claim: split into quality terciles, compare the ROA persistence "
            "slope (next-q ROA on ROA) and the earnings volatility of the best vs worst tercile."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.persistence_by_quality(EV)\n"
            "    gs, ps = q['good_slope'], q['poor_slope']\n"
            "    gev, pev = q['good_earn_vol']*100, q['poor_earn_vol']*100\n"
            "else:\n"
            "    gs, ps = R['good_slope'], R['poor_slope']\n"
            "    gev, pev = R['good_earn_vol'], R['poor_earn_vol']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['low quality','high quality'], [ps, gs], color=[GREY, GREEN], width=.5)\n"
            "for i,v in enumerate([ps, gs]): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('earnings persistence slope'); a1.set_title(f'Persistence: +{gs:.2f} vs +{ps:.2f}')\n"
            "a2.bar(['low quality','high quality'], [pev, gev], color=[GREY, GREEN], width=.5)\n"
            "for i,v in enumerate([pev, gev]): a2.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('earnings volatility (% of assets)'); a2.set_title(f'Earnings vol: {gev:.2f}% vs {pev:.2f}%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'persistence: high-Q {gs:+.3f} vs low-Q {ps:+.3f} (gap {gs-ps:+.3f})  |  '\n"
            "      f'earnings vol: high-Q {gev:.2f}% vs low-Q {pev:.2f}% of assets')"
        ),
        md(
            f"> 💡 In plain words: this is the study's one unambiguous result — high accrual "
            f"quality **means more persistent, less volatile earnings**, with a persistence gap "
            f"of **+{R['slope_gap']:.2f}** (+{R['good_slope']:.2f} vs +{R['poor_slope']:.2f}) and "
            f"~5× lower earnings volatility ({R['good_earn_vol']:.1f}% vs "
            f"{R['poor_earn_vol']:.1f}% of assets). The Dechow-Dichev construct is exactly as "
            "advertised. The return null is therefore not a data problem — it's a "
            "**market-efficiency** result: earnings-quality is a public, slow-moving fundamental "
            "the market already prices. (The persistence *t*'s are pooled across clustered "
            "filings; we cite the magnitude, not the *t*.)"
        ),
        md(
            "### 4e · Tradability — the timer\n\n"
            "For completeness, the calendar long-short net of one-way costs × turnover (both "
            "legs) + short borrow — though a wrong-signed gross *t* already settles it."
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
            "ax.set_title('Wrong-signed before costs; costs only make it worse')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: {a:+.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: turnover is low (~{R['ls_turn']:.2f}/mo — a slow quarterly "
            f"balance-sheet signal), but it doesn't matter: the gross spread was already "
            f"wrong-signed and insignificant, so costs just push net to {R['net'][(20,100)][1]:+.1f}%/yr "
            f"(NW *t* = {R['net'][(20,100)][2]:+.2f}). **Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + quality panel with a TUNABLE planted relation (high-quality names "
            "drift up). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=855 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=855)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted relation (edge=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null barely fires; a planted relation lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 only "
            f"{R['syn_null_fire']}/12 times — about what chance gives you for a HAC *t* over ~180 "
            f"months. A planted relation reads NW *t* = {R['syn_planted_t']:.1f}. The machinery is "
            f"unbiased and powered, so the real-tape {R['ls_t_nw']:+.2f} is a genuine null, not a "
            "broken pipeline. *(Power check only — never cited in support of a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `NONE`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo ({R['ls_ann']:+.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; WC-accrual variant NW *t* = "
            f"{R['wc_t_nw']:+.2f}; pooled event drift negative and non-monotone at every horizon "
            "(placebo *p* 0.73-0.84); both eras flat. Zero, faintly wrong-signed — the desk's "
            "`NONE`.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: "
            f"{R['net'][(20, 100)][1]:+.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20, 100)][3]:.2f}; wrong-signed before costs.\n"
            f"- **Flags earnings quality? `CONFIRMED`** — persistence +{R['good_slope']:.2f} vs "
            f"+{R['poor_slope']:.2f}, earnings vol {R['good_earn_vol']:.1f}% vs "
            f"{R['poor_earn_vol']:.1f}%. The construct is real; the market prices it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The direction worth chasing** is the *information-risk premium*, not the "
            "*discount*: Francis et al. (2005) find poor accrual quality earns a **higher** cost "
            "of capital — the opposite of the retail pitch — and the faint wrong-signed tilt here "
            "is on that side. A public fundamental that grades the business but not the stock is "
            "the textbook signature of an efficiently-priced characteristic; any alpha lives in "
            "the part the market can't already see.\n"
            "- **Coverage honesty:** the rolling 12-quarter window needs ~4 years of clean "
            f"quarterly cash-flow history first, and the cross-section runs from ≈"
            f"{R['xsec_early']:.0f} names (2012) to ≈{R['xsec_late']:.0f} (2024+); terciles on a "
            "thin early cross-section are noisy, and the flat event drift across 2,000+ pooled "
            "events is the more decisive evidence of the return null.\n"
            "- **Dedup map:** [231-sloan-accruals](../../231-sloan-accruals/) (the accrual "
            "*level/sign*), [522-percent-operating-accruals](../../522-percent-operating-accruals/) "
            "(accruals scaled by earnings), [539-cash-flow-volatility](../../539-cash-flow-volatility/) "
            "(raw cash-flow vol — DD nets exactly this out), [52-smoke-screen](../../52-smoke-screen/) "
            "(discretionary-accrual manipulation). None ranks on the DD residual volatility "
            "itself.\n\n"
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
