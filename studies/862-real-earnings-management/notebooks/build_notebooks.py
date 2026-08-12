"""Generate the two narrative notebooks for Study 862 (Real Earnings Management).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR fundamentals + yfinance
# prices, 32 usable manufacturer names, period ends 2009-07 -> 2026-03, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=32, n_roster=44, n_events=630,
    end_lo="2009-07-31", end_hi="2026-03-31", fp_prices="e443c0a5d730",
    # primary calendar long-short (rem, terciles, staleness 200)
    n_months=153, avg_n=17.1, ls_span_lo="2009-11-30", ls_span_hi="2026-03-31",
    ls_mean_bps=-9.2, ls_ann=-1.10, ls_t_iid=-0.21, ls_t_nw=-0.16, ls_sharpe=-0.06,
    ls_hit=46, ls_long_bps=179.6, ls_short_bps=188.8, ls_turn=0.16,
    ls120_mean_bps=75.3, ls120_t_nw=1.82,
    abprod_mean_bps=-5.5, abprod_t_nw=-0.09,
    abdisx_mean_bps=30.7, abdisx_t_nw=0.69,
    # pooled event drift  horizon -> (n, top%, bot%, ls%, t, win%, placebo p)
    drift={
        21: (630, 1.70, 1.26, 0.44, 0.63, 46, 0.5308),
        63: (629, 4.65, 5.52, -0.87, -0.62, 47, 0.4903),
        126: (628, 10.16, 10.45, -0.28, -0.14, 45, 0.8796),
    },
    mono63=(5.52, 3.39, 4.65),
    # era split (calendar LS), split 2015
    era_early_n=62, era_early_bps=69.5, era_early_t=1.02,
    era_late_n=91, era_late_bps=-62.8, era_late_t=-0.79,
    # leads operating (gross-margin reversal)
    op_n=339, op_slope=0.0569, op_t=1.36, op_r2=0.005, op_corr=0.074,
    op_top_dgm=0.56, op_bot_dgm=0.08, op_spread=0.48,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (-9.0, -1.08, -0.16, -0.06), (20, 100): (-8.8, -1.05, -0.16, -0.06)},
    # synthetic control
    syn_null_mean=0.31, syn_null_sd=0.78, syn_null_fire=0, syn_null_seeds=12,
    syn_planted_bps=874.2, syn_planted_t=20.28,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Operating reversal%3F: Not detected](https://img.shields.io/badge/Operating_reversal%3F-Not_detected-8b949e?style=flat-square)\n\n"
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

from real_earn_mgmt import data, strategy as st

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
            "# Companies really do cook earnings without touching the books. Can you trade it? 🎭\n"
            "### 'Real earnings management' — overproduce and slash R&D to hit the number — is a "
            "genuine forensic tell. As a stock signal it's a coin-flip.\n\n"
            + BADGES +
            "When a company is about to *miss* the number Wall Street expects, it has two ways to "
            "fudge it. The famous one is **accrual** management — accounting choices about when to "
            "book things. The sneakier one, and the subject here, is **real** earnings management "
            "(Roychowdhury 2006): change the *actual business* for a quarter. **Overproduce** — run "
            "the factory hot so fixed costs spread over more units and the reported cost of each "
            "sale drops. **Cut discretionary spend** — freeze R&D, trim SG&A, kill the ad budget. "
            "Both flatter this quarter's profit, and both leave a fingerprint: production looks "
            "*abnormally high*, discretionary expense *abnormally low*.\n\n"
            "The forensic idea is real and well-documented. The tempting leap — *so buy the "
            "manipulators, or short them* — is where it falls apart.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 44 large US manufacturers / hardware / pharma / industrials that "
            "disclose the needed line items on EDGAR (32 usable), 2009→2026; a genuinely **thin, "
            "uneven panel**. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is 'real earnings management' a real thing? | **Yes.** Firms demonstrably "
            "overproduce and cut discretionary spend to hit targets — it's one of the most-cited "
            "results in accounting. |\n"
            f"| Does the fingerprint predict **stock returns**? | **No.** A long-short that ranks "
            f"firms on the real-management proxy earns **{R['ls_mean_bps']:+.0f} bps/month** — a "
            f"robust *t* of **{R['ls_t_nw']:+.2f}**, an unmistakable zero. |\n"
            "| Is it at least *stably* zero? | **Not even that.** Nudge one knob (how stale a "
            f"signal you'll hold) and the sign flips from −9 to **+75 bps/mo**; cut the sample at "
            f"2015 and it's **+70 bps early, −63 late**. A real edge doesn't flip sign when you "
            "sneeze on it. |\n"
            "| Can you trade it? | **No.** There is nothing to charge costs against. |\n\n"
            "> A great forensic tell about the *business* is not automatically a signal about the "
            "*stock*. That distinction is the whole study."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Firms about to miss earnings overproduce (to bury fixed costs in inventory) and "
            "cut R&D / SG&A. That leaves abnormally high production and abnormally low "
            "discretionary expense — spot it, and you've spotted a manipulator whose stock will "
            "eventually pay for it.\"*\n\n"
            "Roychowdhury (2006) built exactly this: fit a 'normal' level of production and of "
            "discretionary expense from ordinary firms, and call the leftover — the **residual** — "
            "the *abnormal*, real-management piece. The follow-up literature argues real "
            "management is **value-destroying** (you discount the excess inventory later; you "
            "starve the R&D that drives growth), so the manipulators should *under*-perform. We "
            "test whether that shows up in tradeable returns."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a residual you can compute from any 10-Q flagged tomorrow's losers, it would be a "
            "gift: no insider data, just cost of goods sold and R&D. That is precisely why to be "
            "suspicious. Everyone can read the same filing; if the market prices public forensic "
            "signals at all, this is the kind it should price fast — and a 'signal' that only "
            "*sometimes* has a sign is usually noise wearing a lab coat."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** For each of {R['n_names']} manufacturers we fit Roychowdhury's two "
            "'normal' models and take the residuals: **abnormal production** (high = "
            "overproducing) and **abnormal discretionary expense** (low = cutting). Add them into "
            "one **REM** score, known only on the **filing date** (no peeking).\n"
            "- **The return test.** Each month, rank the names on REM, buy the top third (most "
            "real management), short the bottom, hold a month. Does the spread make money — and "
            "can we tell it from noise?\n"
            "- **The robustness test.** Does the answer survive changing the staleness window and "
            "splitting the sample in half? (Spoiler: the sign doesn't.)\n"
            "- **The mechanism test.** If firms overproduced to flatter margins, next quarter's "
            "gross margin should *fall back*. Does it?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The money question first: do the stocks sort?** Same monthly ranking; we measure "
            "the forward *return* of a buy-the-top-third (manipulators), short-the-bottom-third "
            "(clean) portfolio and plot the growth of \\$1."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='rem', n_buckets=3, min_names=6, staleness_days=200)\n"
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
            "    ax.set_title(f'A wandering line that means nothing: {ann:+.1f}%/yr, robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: {ann:+.1f}%/yr gross, Newey-West t = {tnw:+.2f} (the bar is |2|)')"
        ),
        md(
            f"It wanders around \\$1 and ends slightly **below** it — **{R['ls_ann']:+.1f}%/yr** "
            f"gross, robust *t* = **{R['ls_t_nw']:+.2f}**. That is a zero. Not 'a small edge': a "
            "coin-flip whose cumulative line drifts because monthly returns are noisy, not because "
            "anything is being predicted.\n\n"
            "**The tell that it's noise:** a real signal keeps its sign when you change an "
            "arbitrary knob. This one doesn't. Here's the same long-short computed two ways — "
            "holding a signal up to ~6.5 months (staleness 200) vs ~4 months (staleness 120) — "
            "and split into an early and a late half."
        ),
        code(
            "labels = ['staleness 200d','staleness 120d','2010-2014','2015-2026']\n"
            "if HAVE_REAL:\n"
            "    ls200 = st.calendar_ls(PX, EV, signal_col='rem', staleness_days=200)\n"
            "    ls120 = st.calendar_ls(PX, EV, signal_col='rem', staleness_days=120)\n"
            "    b200 = st.calendar_ls_stats(ls200)['mean_bps']; b120 = st.calendar_ls_stats(ls120)['mean_bps']\n"
            "    e = ls200[ls200.index < '2015-01-01']['ls'].to_numpy(); l = ls200[ls200.index >= '2015-01-01']['ls'].to_numpy()\n"
            "    vals = [b200, b120, e.mean()*1e4, l.mean()*1e4]\n"
            "else:\n"
            "    vals = [R['ls_mean_bps'], R['ls120_mean_bps'], R['era_early_bps'], R['era_late_bps']]\n"
            "cols = [RED if v < 0 else AMBER for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f} bps',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.9); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Same signal, opposite signs: the fingerprint of noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('the long-short mean flips sign across the staleness knob and across eras — not an edge')"
        ),
        md(
            f"Look at that: **{R['ls_mean_bps']:+.0f}** vs **{R['ls120_mean_bps']:+.0f}** bps just "
            f"from the staleness knob, and **{R['era_early_bps']:+.0f}** vs "
            f"**{R['era_late_bps']:+.0f}** across the 2015 split. A live edge is stubborn about its "
            "sign; this one changes its mind every time we change an arbitrary setting. That's the "
            "signature of no signal at all.\n\n"
            "**And the ladder?** If REM sorted returns, the middle third should sit between the "
            "top and bottom. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=63)\n"
            "    mono = st.bucket_means(fr, 3)*100\n"
            "else:\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['bottom third\\n(clean)','middle\\nthird','top third\\n(manipulators)'], mono,\n"
            "       color=[GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 3-month forward return')\n"
            "ax.set_title('No ladder: the middle third is the WORST (non-monotone)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('3-month forward return by REM third (low->high):', [f'{v:+.2f}%' for v in mono])"
        ),
        md(
            "No staircase — the *middle* third is the worst, which no real sort produces. Whether "
            "we look at the calendar long-short, the sign-instability, or this event-drift ladder, "
            "the three agree: **there is no return sort in the REM proxy on this tape.**\n\n"
            "**One more, the mechanism:** if the overproduction story were operating here, high-REM "
            "firms' gross margins should *give back* next quarter. Do they?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.leads_operating(EV)\n"
            "    top, bot = q['top_dgm']*100, q['bot_dgm']*100\n"
            "else:\n"
            "    top, bot = R['op_top_dgm'], R['op_bot_dgm']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(['bottom third\\n(clean)','top third\\n(manipulators)'], [bot, top], color=[GREY, GREY], width=.5)\n"
            "for i,v in enumerate([bot, top]): ax.annotate(f'{v:+.2f} pp',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel(\"next quarter's gross-margin CHANGE\")\n"
            "ax.set_title('No reversal either: high-REM margins do not fall back (wrong sign, tiny)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'forward gross-margin change: top third {top:+.2f} pp vs bottom {bot:+.2f} pp — no give-back')"
        ),
        md(
            "The manipulator third's gross margin does **not** fall back next quarter — if anything "
            "it edges *up*, the opposite of the reversal the story predicts, and by a statistically "
            "negligible amount. On this small, industry-pooled panel the operating mechanism is not "
            "observable either. (That's an honest *non-detection*, not a claim that Roychowdhury's "
            "carefully-built industry-year original was wrong.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) — None.** Robust *t* = {R['ls_t_nw']:+.2f}, and the sign flips "
            "across the staleness knob and the 2015 split. The event drift is flat and "
            "non-monotone. No stable sign, no significance.\n"
            "- **Tradability — Mirage.** You can't get paid for a coin-flip; it fails before costs.\n"
            "- **Operating reversal — Not detected.** High-REM firms show no next-quarter "
            "gross-margin give-back on this panel (wrong-signed, insignificant).\n\n"
            "> The honest one-liner: *real earnings management is a genuine thing companies do, "
            "and a genuinely useless thing to rank stocks on — here, twice over.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where an edge might still hide.** Not in a pooled residual across mixed "
            "industries, but in Roychowdhury's original **industry-year** estimation with hundreds "
            "of names per bin, or **conditioned on firms that just barely beat** consensus (the "
            "population REM was designed to flag). A cleaner residual on a wider panel is the "
            "honest next step — this thin, pooled version can't see it.\n"
            "- **The coverage caveat is real.** Quarterly Q4 tags are missing, several names never "
            f"report R&D, and only {R['n_names']} of {R['n_roster']} names clear the usable bar — "
            "the cross-section averages ≈17. A deeper panel might sharpen the test, though the flat "
            "event drift makes a large hidden return edge unlikely.\n"
            "- **Sibling studies:** [574-penny-beat](../../574-penny-beat/) (the *just-beat* "
            "discontinuity REM is one mechanism for), [229-beneish-m-score](../../229-beneish-m-score/) "
            "and [855-accrual-quality](../../855-accrual-quality/) (the *accrual* channel REM "
            "avoids), and [525-r-and-d-intensity](../../525-r-and-d-intensity/) (the R&D *level*, "
            "not the residual). See [docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the alpha is in the industry-year residual or the just-beat subsample? Build "
            "it, show a certifiable net spread on the size you'd actually run — then we'll talk.*"
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
            "# Real Earnings Management — a quantitative teardown 🔬\n"
            "### Roychowdhury's two normal-level residuals · a point-in-time calendar-time tercile "
            "long-short (Newey-West) · sign-instability across staleness & eras · a pooled "
            "event-drift cross-check with a two-sided label-shuffle placebo · the gross-margin-"
            "reversal regression · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — a **Roychowdhury (2006) REM proxy** (abnormally high production + abnormally "
            "low discretionary expense) **predicts forward returns** — resolves into a clean null "
            "that isn't even sign-stable. This is distinct from every sibling on the desk: "
            "[574](../../574-penny-beat/) documents the *just-beat* discontinuity, "
            "[229](../../229-beneish-m-score/) and [855](../../855-accrual-quality/) detect "
            "*accrual* manipulation, [525](../../525-r-and-d-intensity/) ranks the R&D *level*. "
            "This is the **real-activity** residual.\n\n"
            "> ⚠️ **Data note.** EDGAR `Revenues`/`CostOfRevenue`/`SG&A`/`R&D` (flows) + "
            "`InventoryNet`/`Assets` (instants) + yfinance closes, "
            + str(R["n_names"]) + " usable names, ends "
            + R["end_lo"] + " → " + R["end_hi"] + ", as-of " + R["as_of"] + ". Point-in-time on "
            "the **filing date**. Survivorship named on the Signal axis (current-survivors "
            "basket). Thin/uneven coverage + pooled-industry, in-sample benchmark are first-class "
            "caveats. Numbers in [`docs/results.md`](../docs/results.md) (prices fingerprint `"
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
            f"*t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; sign flips to "
            f"+{R['ls120_mean_bps']:.0f} bps at staleness 120 and across the 2015 era split |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: NW *t* = "
            f"{R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; a zero gross "
            "edge |\n"
            f"| **Operating reversal?** | `NOT DETECTED` | forward Δ gross-margin on REM: slope "
            f"+{R['op_slope']:.3f} (*t* +{R['op_t']:.2f}, R² {R['op_r2']:.3f}), wrong-signed vs the "
            "reversal prediction |\n\n"
            "> 💡 In plain words: a beautiful forensic construct that neither sorts returns nor, on "
            "this pooled panel, shows its own operating footprint. A public residual the market has "
            "no need to misprice."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "Scale everything by lagged assets $A_{t-1}$. Roychowdhury's two normal-level models, "
            "fit cross-sectionally:\n\n"
            "$$\\frac{DISX_t}{A_{t-1}} = a_0 + a_1\\frac{1}{A_{t-1}} + a_2\\frac{Sales_{t-1}}"
            "{A_{t-1}} + \\varepsilon^{DISX}_t,\\qquad DISX = R\\&D + SG\\&A$$\n\n"
            "$$\\frac{PROD_t}{A_{t-1}} = a_0 + a_1\\frac{1}{A_{t-1}} + a_2\\frac{Sales_t}{A_{t-1}} "
            "+ a_3\\frac{\\Delta Sales_t}{A_{t-1}} + a_4\\frac{\\Delta Sales_{t-1}}{A_{t-1}} + "
            "\\varepsilon^{PROD}_t,\\qquad PROD = COGS + \\Delta Inv$$\n\n"
            "The abnormal pieces are the residuals; $\\text{REM} = \\varepsilon^{PROD} - "
            "\\varepsilon^{DISX}$ (overproduce **and** cut discretionary ⇒ high REM). The claims:\n\n"
            "- **H₁ (leads returns).** A cross-sectional long-short on REM earns a forward spread "
            "(the mispricing claim; the literature's sign is *negative* — manipulators "
            "under-perform).\n"
            "- **H₂ (operating reversal).** High REM foreshadows a **drop** in next-quarter gross "
            "margin (overproduction give-back).\n"
            "- **H₃ (tradable).** Any spread survives realistic long-short costs + borrow.\n\n"
            f"We find **H₁ a flat null** (NW *t* = {R['ls_t_nw']:+.2f}, sign-unstable), **H₂ not "
            f"detected** (slope +{R['op_slope']:.3f}, wrong-signed, *t* +{R['op_t']:.2f}), and "
            "therefore **H₃ moot**. The DGP the sort would need is simply not in this tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, because the "
            "signal is persistent and filings cluster: a monthly long-short series lets a "
            "**Newey-West (6-lag) HAC *t*** do the honest work the desk's `REAL` bar is written "
            "against. The panel is thin, so we sort into **terciles** (not quintiles) and require "
            "≥ 6 names. Crucially we also probe **sign-stability** — a real edge keeps its sign "
            "when the staleness window or the sample split changes; noise doesn't. The pooled event "
            "drift + a **two-sided** label-shuffle placebo is the cross-check (two-sided because "
            "the sign is a hypothesis); the gross-margin-reversal regression is the mechanism axis, "
            "read on magnitude (its pooled *t* ignores quarter clustering)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) firm-quarters across {R['n_names']} "
            f"usable names (of {R['n_roster']} in the roster), ends {R['end_lo']} → {R['end_hi']}, "
            "each stamped with its 10-Q/10-K filing date (point-in-time).\n"
            "- **Signal.** Pooled OLS residuals of the two normal-level models → `ab_prod`, "
            "`ab_disx`; `rem = ab_prod − ab_disx`.\n"
            "- **Primary.** Monthly tercile long-short on `rem`, one execution lag; Newey-West + "
            "one-sample *t*, Sharpe, hit rate.\n"
            "- **Robustness.** Staleness 120 vs 200 d; the two components separately; an era split "
            "at 2015.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag entry, "
            "top-minus-bottom tercile, one-sample *t* + two-sided label-shuffle placebo, and the "
            "tercile monotonicity.\n"
            "- **Mechanism.** Pooled OLS of the forward gross-margin change on `rem`.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short borrow.\n"
            "- **Control.** Synthetic panel, planted-lead knob; the null must not fire across 12 "
            "seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh REM signals into terciles each month, long top / short bottom equal-weight, "
            "earn next month's return. The decisive statistic is the HAC *t* of the monthly series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='rem', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    lsp = st.calendar_ls(PX, EV, signal_col='ab_prod', staleness_days=200); sp = st.calendar_ls_stats(lsp)\n"
            "    lsd = st.calendar_ls(PX, EV, signal_col='ab_disx', staleness_days=200); sd = st.calendar_ls_stats(lsd)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo ({s['ann_pct']:+.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  component ab_prod: {sp['mean_bps']:+.1f} bps, NW t = {sp['t_nw']:+.2f}  |  \"\n"
            "          f\"ab_disx: {sd['mean_bps']:+.1f} bps, NW t = {sd['t_nw']:+.2f}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=RED, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: {R['ls_ann']:+.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven: the usable panel averages ~17 names')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}; \"\n"
            "      f\"ab_prod NW t = {R['abprod_t_nw']:+.2f}, ab_disx NW t = {R['abdisx_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **{R['ls_mean_bps']:+.0f} bps/month** at HAC *t* = "
            f"**{R['ls_t_nw']:+.2f}** is a zero. And the two ingredients don't even agree — "
            f"`ab_prod` alone is {R['abprod_mean_bps']:+.1f} bps (NW *t* = {R['abprod_t_nw']:+.2f}), "
            f"`ab_disx` alone is {R['abdisx_mean_bps']:+.1f} bps (NW *t* = {R['abdisx_t_nw']:+.2f}). "
            f"With n={R['n_months']} months and an avg cross-section of {R['avg_n']:.0f}, the sort "
            "carries no return information."
        ),
        md(
            "### 4b · Sign-instability — the giveaway\n\n"
            "A real edge is stubborn about its sign. Recompute the same long-short with a shorter "
            "staleness window, and split it at 2015."
        ),
        code(
            "labels = ['staleness\\n200d','staleness\\n120d','2010-2014','2015-2026']\n"
            "if HAVE_REAL:\n"
            "    ls200 = st.calendar_ls(PX, EV, signal_col='rem', staleness_days=200)\n"
            "    ls120 = st.calendar_ls(PX, EV, signal_col='rem', staleness_days=120)\n"
            "    b200, b120 = st.calendar_ls_stats(ls200)['mean_bps'], st.calendar_ls_stats(ls120)['mean_bps']\n"
            "    e = ls200[ls200.index < '2015-01-01']['ls'].to_numpy(); l = ls200[ls200.index >= '2015-01-01']['ls'].to_numpy()\n"
            "    vals = [b200, b120, e.mean()*1e4, l.mean()*1e4]\n"
            "    ts   = [st.calendar_ls_stats(ls200)['t_nw'], st.calendar_ls_stats(ls120)['t_nw'], st.newey_west_t(e), st.newey_west_t(l)]\n"
            "else:\n"
            "    vals = [R['ls_mean_bps'], R['ls120_mean_bps'], R['era_early_bps'], R['era_late_bps']]\n"
            "    ts   = [R['ls_t_nw'], R['ls120_t_nw'], R['era_early_t'], R['era_late_t']]\n"
            "cols = [RED if v < 0 else AMBER for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(labels, vals, color=cols, width=.62)\n"
            "for i,(v,t_) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.0f} bps\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.9); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Same signal, opposite signs across a knob and a date — noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('staleness/era cuts flip the sign; a live edge would not')"
        ),
        md(
            f"> 💡 In plain words: {R['ls_mean_bps']:+.0f} bps (staleness 200) becomes "
            f"**{R['ls120_mean_bps']:+.0f}** bps (staleness 120), and {R['era_early_bps']:+.0f} bps "
            f"pre-2015 becomes {R['era_late_bps']:+.0f} bps post. None clears |2|. A signal whose "
            "sign is a coin-flip of the researcher's arbitrary choices is the textbook picture of "
            "**no effect** — this is what earns the `NONE`, not merely a sub-2 *t*."
        ),
        md(
            "### 4c · The cross-check — pooled event drift + two-sided placebo + monotonicity\n\n"
            "Bucket all events by REM; top-minus-bottom forward drift with a two-sided label-"
            "shuffle null. A real sort would form a ladder."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for h in st.HORIZONS:\n"
            "        es = st.event_summary(PX, EV, horizon=h, n_buckets=3, n_draws=2000)\n"
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
            "a1.set_ylabel('one-sample t (long-short drift)'); a1.set_title('Flat & sign-flipping: no horizon clears |t|=2')\n"
            "a2.bar(['bottom','middle','top'], mono, color=GREY, width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('No ladder (middle third is worst)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift **flips sign across horizons** "
            f"({R['drift'][21][3]:+.2f}% at 21d, {R['drift'][63][3]:+.2f}% at 63d), no *t* exceeds "
            f"±0.63, the two-sided placebo *p* sits at **{R['drift'][63][6]:.2f}–{R['drift'][126][6]:.2f}** "
            f"(a random split matches the real one about half the time), and the terciles are "
            f"**non-monotone** — the *middle* third ({R['mono63'][1]:+.2f}% at 63d) is the worst. "
            "The event study and the calendar long-short agree: no return sort."
        ),
        md(
            "### 4d · The mechanism — no gross-margin reversal either\n\n"
            "Pooled OLS of the forward gross-margin change ($next\\_gm - gm$) on REM, and the "
            "forward-Δgm spread between the top and bottom REM terciles. The value-destruction "
            "story predicts a **negative** slope."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.leads_operating(EV)\n"
            "    fr = EV.dropna(subset=['rem','gm','next_gm'])\n"
            "    x = fr['rem'].to_numpy(); y = (fr['next_gm']-fr['gm']).to_numpy()\n"
            "    slope, corr, top, bot = q['slope'], q['corr'], q['top_dgm']*100, q['bot_dgm']*100\n"
            "else:\n"
            "    x = y = None\n"
            "    slope, corr, top, bot = R['op_slope'], R['op_corr'], R['op_top_dgm'], R['op_bot_dgm']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if x is not None:\n"
            "    m = (np.abs(x)<np.percentile(np.abs(x),98))&(np.abs(y)<0.3)\n"
            "    a1.scatter(x[m], y[m]*100, s=9, alpha=.3, color=GREY)\n"
            "    xs = np.linspace(np.percentile(x,2), np.percentile(x,98), 50)\n"
            "    a1.plot(xs, (slope*xs + (y.mean()-slope*x.mean()))*100, color=RED, lw=2)\n"
            "    a1.set_xlabel('REM'); a1.set_ylabel('NEXT-quarter gross-margin change (pp)')\n"
            "    a1.set_title(f'No reversal: slope {slope:+.3f} (wrong sign), corr {corr:+.2f}')\n"
            "else:\n"
            "    a1.text(.5,.5,'run with cache',ha='center'); a1.set_axis_off()\n"
            "a2.bar(['bottom third\\n(clean)','top third\\n(manipulators)'], [bot, top], color=[GREY, GREY], width=.5)\n"
            "for i,v in enumerate([bot, top]): a2.annotate(f'{v:+.2f} pp',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('next-q gross-margin change'); a2.set_title('No give-back in the top-REM tercile')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'leads-operating: slope {slope:+.4f}, corr {corr:+.3f}, forward-dgm spread {top-bot:+.2f} pp (top {top:+.2f} vs bottom {bot:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the operating footprint is **absent** here — slope "
            f"+{R['op_slope']:.3f} (*t* +{R['op_t']:.2f}, R² {R['op_r2']:.3f}), the **wrong sign** "
            f"for a reversal, and a forward-Δgm spread of only +{R['op_spread']:.2f} pp. Unlike a "
            "study where the mechanism holds but the market prices it, here neither the return sort "
            "nor the mechanism appears — consistent with the caveat that a *pooled-industry*, "
            "*in-sample* residual on ~17 names is a noisy proxy for Roychowdhury's industry-year "
            "construct. An honest non-detection, not a refutation."
        ),
        md(
            "### 4e · Tradability — the timer\n\n"
            "For completeness, the calendar long-short net of one-way costs × turnover (both legs) "
            "+ short borrow — though a zero gross edge already settles it."
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
            "ax.set_title('Negative net of costs — but the gross edge was already zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: {a:+.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: net NW *t* = {R['net'][(20,100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20,100)][3]:.2f}. There was nothing to cost — the gross edge is a "
            "(slightly negative) zero. **Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + REM-signal panel with a TUNABLE planted lead (high-REM names drift "
            "up). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=862 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=862)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted lead (edge=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null never fires; a planted lead lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 "
            f"{R['syn_null_fire']}/12 times; a planted lead reads NW *t* = {R['syn_planted_t']:.1f}. "
            f"The machinery is unbiased and powered, so the real-tape {R['ls_t_nw']:+.2f} is a "
            "genuine null, not a broken pipeline. *(Power check only — never cited in support of a "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `NONE`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo ({R['ls_ann']:+.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; the sign **flips** across the staleness knob "
            f"(+{R['ls120_mean_bps']:.0f} bps at 120 d) and the 2015 split "
            f"({R['era_early_bps']:+.0f} / {R['era_late_bps']:+.0f} bps); the pooled event drift is "
            "flat, sign-flipping and non-monotone (placebo *p* ≈ 0.5–0.9). No stable sign, no "
            "certification.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: {R['net'][(20, 100)][1]:+.2f}"
            f"%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; a "
            "zero gross edge.\n"
            f"- **Operating reversal `NOT DETECTED`** — forward Δ gross-margin on REM: slope "
            f"+{R['op_slope']:.3f} (*t* +{R['op_t']:.2f}, R² {R['op_r2']:.3f}), wrong-signed and "
            "insignificant. On this pooled-industry panel the mechanism is not observable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The residual worth chasing** is Roychowdhury's original **industry-year** "
            "estimation (hundreds of names per bin, so the residual isn't industry-contaminated), "
            "and the population REM was designed for: firms that **just barely beat** the "
            "benchmark. A cleaner residual on a wider panel is the honest next step — the flat, "
            "sign-unstable result here is exactly what a noisy pooled proxy on ~17 names should "
            "produce under the null.\n"
            f"- **Coverage honesty:** only {R['n_names']} of {R['n_roster']} names clear the usable "
            "bar (Q4 flow tags missing, several names never report R&D), the cross-section averages "
            "≈17, and the benchmark coefficients are pooled in-sample across industries. Every "
            "number should be read in that light.\n"
            "- **Dedup map:** [574-penny-beat](../../574-penny-beat/) (the *just-beat* "
            "discontinuity), [229-beneish-m-score](../../229-beneish-m-score/) and "
            "[855-accrual-quality](../../855-accrual-quality/) (the *accrual* channel), "
            "[525-r-and-d-intensity](../../525-r-and-d-intensity/) (the R&D *level*). None ranks on "
            "the Roychowdhury REM residual itself.\n\n"
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
