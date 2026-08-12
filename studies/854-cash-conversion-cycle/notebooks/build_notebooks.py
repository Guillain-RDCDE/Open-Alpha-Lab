"""Generate the two narrative notebooks for Study 854 (Cash Conversion Cycle).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR CCC components +
# yfinance prices, 30 large US filers with a matched five-fact history, period ends
# 2009-07-31 → 2026-05-02, as-of 2026-06-30). Reproduce with examples/verify.py.
R = dict(
    as_of="2026-06-30", n_names=30, n_events=689,
    end_lo="2009-07-31", end_hi="2026-05-02", fp_prices="d82e55a0af11",
    n_months=190, avg_n=13.9, ls_span_lo="2010-09-30", ls_span_hi="2026-06-30",
    ls_mean_bps=-47.1, ls_ann=-5.65, ls_t_iid=-1.37, ls_t_nw=-1.03, ls_sharpe=-0.34,
    ls_hit=47, ls_hit_lo=40, ls_hit_hi=54, ls_long_bps=125.1, ls_short_bps=172.2,
    ls_turn=0.17, ls_cum=0.33,
    ls120_mean_bps=-70.5, ls120_t_nw=-1.45,
    pct_mean_bps=-2.8, pct_t_nw=-0.07, pct_sharpe=-0.02,
    xsec_early=16.4, xsec_late=11.7,
    drift={21: (687, 1.25, 1.00, 0.25, 0.40, 52, 0.3379),
           63: (681, 4.26, 4.59, -0.33, -0.27, 53, 0.6175),
           126: (676, 8.43, 8.19, 0.24, 0.09, 52, 0.4562)},
    mono63=(4.59, 3.33, 4.26),
    era_early_n=88, era_early_bps=31.9, era_early_t=0.84,
    era_late_n=102, era_late_bps=-115.3, era_late_t=-1.54,
    im_n=487, im_slope=-0.00010, im_t=-1.13, im_r2=0.003, im_corr=-0.051,
    im_low=0.0040, im_high=0.0013, im_spread=0.0027,
    net={(10, 50): (-54.6, -6.55, -1.19, -0.40), (20, 100): (-62.1, -7.45, -1.36, -0.45)},
    syn_null_mean=0.18, syn_null_sd=0.59, syn_null_fire=0, syn_null_seeds=12,
    syn_planted_bps=907.2, syn_planted_t=18.07,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Precedes margin%3F: Faint](https://img.shields.io/badge/Precedes_margin%3F-Faint-8b949e?style=flat-square)\n\n"
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

from ccc_signal import data, strategy as st

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
            "# A shorter Cash Conversion Cycle frees cash. Does the stock care? 🔄\n"
            "### A textbook working-capital efficiency signal — does squeezing the cash cycle "
            "actually predict returns, or just tell you what good operators already look like?\n\n"
            + BADGES +
            "Every operations textbook and every CFO working-capital scorecard sells the same "
            "idea. A company's **Cash Conversion Cycle** is how many days its money sits trapped "
            "in the business between paying suppliers and getting paid by customers:\n\n"
            "$$\\text{CCC} = \\underbrace{\\text{DSO}}_{\\text{days to collect}} + "
            "\\underbrace{\\text{DIO}}_{\\text{days of inventory}} - "
            "\\underbrace{\\text{DPO}}_{\\text{days to pay suppliers}}$$\n\n"
            "Collect faster, hold less stock, stretch your payables, and the number falls — you've "
            "freed cash you can put to work. The pitch writes itself: firms that **shorten** their "
            "CCC are getting more efficient and should out-earn; firms whose CCC is **bloating** "
            "are a slow-motion working-capital drag. So rank on the change and trade it.\n\n"
            "It's a genuinely good operating metric. Whether it's a *stock* signal is a different "
            "question — and that's the one we answer.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** ~46 large US filers that carry real inventory and payables "
            "(consumer, retail, industrials, healthcare-products, hardware, materials) reporting "
            "all five CCC ingredients on EDGAR; a genuinely **thin, uneven panel** (the CCC needs "
            "five matched facts a quarter, so only 30 names survive with a long clean history). "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a **shortening** CCC predict **higher future returns**, as the efficiency "
            f"story claims? | **No — if anything the reverse, and inside the noise.** Long the "
            f"shorteners / short the bloaters earns **{R['ls_ann']:+.1f}%/yr** gross, robust *t* = "
            f"**{R['ls_t_nw']:+.2f}** — the falling-CCC names actually *underperform* the rising-CCC "
            f"ones. And the sign **flips across eras** (*t* = +{R['era_early_t']:.2f} pre-2018, "
            f"{R['era_late_t']:+.2f} post-2018): the fingerprint of noise. |\n"
            f"| Does shortening the CCC at least **precede a better margin**? | **Faintly, and "
            f"right-signed.** Shortening names see next quarter's gross margin widen "
            f"+{R['im_spread']*100:.2f} points more than bloating names — correctly signed but a "
            f"whisper (correlation {R['im_corr']:+.2f}). |\n"
            f"| Can you trade it? | **No.** Wrong-signed and insignificant *before* costs; after "
            f"realistic costs + borrow it is **{R['net'][(20, 100)][1]:+.1f}%/yr**. |\n"
            "| Why so weak? | Five accounting lines anyone can pull from a 10-Q, on **big, liquid, "
            "heavily-covered** names — the last place a durable edge survives. Here it didn't "
            "survive at all. |\n\n"
            "> A textbook working-capital virtue that, on blue chips, sorts neither the stock "
            "(wrong-signed noise) nor much the margin it's built from."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The cash conversion cycle is how long your money is locked up in operations. "
            "Shorten it and you free cash to grow or return; let it bloat and you're financing a "
            "swelling pile of receivables and inventory. So buy the firms shrinking their CCC and "
            "short the ones expanding it.\"*\n\n"
            "It's a respectable idea with real academic backing — a long operations-finance "
            "literature (Richards-Laughlin 1980; Deloof 2003) ties a shorter CCC to higher "
            "profitability. The question is whether that operating truth is also a *trading* edge, "
            "or a fact the market reads off the same 10-Q you do."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a number you can compute from three balance-sheet lines and two income-statement "
            "lines predicted stock returns, it would be one of the easiest anomalies going — no "
            "estimates feed, no alt-data, just arithmetic on every quarterly filing. That's "
            "exactly why it deserves suspicion: it's *too* readable. Everyone can compute a cash "
            "conversion cycle. If markets price public accounting information at all, this is the "
            "kind they should price fast."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The signal.** For each name, the year-over-year change in CCC = DSO + DIO − DPO, "
            "known only on the **filing date** of the 10-Q/10-K (never before — no peeking at a "
            "number that isn't public yet).\n"
            "- **The margin test.** Does a *shrinking* CCC this quarter precede a *wider* gross "
            "margin next quarter? (The 'frees cash → out-earns' mechanism.)\n"
            "- **The return test.** Each month, rank the names on the signal, buy the shortening "
            "third, short the bloating third, hold for the next month. Does the spread make money "
            "— and can we tell it apart from noise?\n"
            "- **The mirage check.** If the return spread can't beat a coin-flip relabelling of "
            "the names, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the mechanism: does a shrinking CCC precede a better margin?** For every "
            "filing we line up this quarter's CCC change against the *next* quarter's change in "
            "gross margin, and split the names into thirds."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.improves_margin(EV)\n"
            "    lo, hi, corr = q['low_margin']*100, q['high_margin']*100, q['corr']\n"
            "else:\n"
            "    lo, hi, corr = R['im_low']*100, R['im_high']*100, R['im_corr']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['bloating third\\n(rising CCC)','shortening third\\n(falling CCC)'],\n"
            "       [hi, lo], color=[GREY, GREEN], width=.55)\n"
            "for i,v in enumerate([hi, lo]): ax.annotate(f'{v:+.2f}pp',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel(\"next quarter's gross-margin change (pp)\")\n"
            "ax.set_title(f'Does shortening the cash cycle precede a better margin? (corr {corr:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'next-q margin change: shortening third {lo:+.2f}pp vs bloating third {hi:+.2f}pp '\n"
            "      f'-> spread {lo-hi:+.2f} pp, correlation {corr:+.3f}')"
        ),
        md(
            f"So there's a *nudge* the right way: shortening-CCC names see next quarter's gross "
            f"margin widen about **+{R['im_spread']*100:.2f} points** more than bloating names, and "
            f"the slope is negative as the 'frees cash → out-earns' story wants. But look how "
            f"*small* it is — correlation **{R['im_corr']:+.2f}**, essentially a flat scatter. The "
            "mechanism is there in the *sign* and almost vanishes in the *magnitude*. Hold that "
            "thought — because it's the strongest thing in this whole study.\n\n"
            "**Now the money question: do the stocks follow?** Same ranking, but instead of future "
            "margin we measure the forward *return* of a buy-the-shortening-third, "
            "short-the-bloating-third portfolio."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='ccc_score', n_buckets=3, min_names=6, staleness_days=200)\n"
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
            "    ax.set_title(f'{ann:+.1f}%/yr on paper (wrong way), robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: {ann:+.1f}%/yr gross, Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"It doesn't drift up — it drifts **down**: **{R['ls_ann']:+.1f}%/yr** gross, turning "
            f"\\$1 into just **\\${R['ls_cum']:.2f}** over {R['n_months']} months. The long leg "
            f"(falling CCC) actually *under*-earns the short leg (rising CCC) — the exact opposite "
            f"of the claim — and the robust *t* is **{R['ls_t_nw']:+.2f}**, well inside the noise "
            f"band around zero. Worse for the story, the sign isn't even stable: it's mildly "
            f"*positive* before 2018 (*t* = +{R['era_early_t']:.2f}) and mildly *negative* after "
            f"(*t* = {R['era_late_t']:+.2f}). A 'signal' whose sign depends on the decade isn't a "
            "signal.\n\n"
            "**The tell:** if the CCC change really sorted returns, the thirds would climb from "
            "bloating to shortening. They don't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=63)\n"
            "    mono = st.bucket_means(fr, 3)*100\n"
            "else:\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['bloating\\nthird','middle\\nthird','shortening\\nthird'], mono,\n"
            "       color=[GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 3-month forward return')\n"
            "ax.set_title('The return thirds by CCC change (bloating -> shortening)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('3-month forward return by CCC-change third (bloating->shortening):', [f'{v:+.2f}%' for v in mono])"
        ),
        md(
            f"No staircase — the thirds come in **{R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / "
            f"{R['mono63'][2]:+.2f}%** (bloating → shortening): a dip in the middle, not a ladder, "
            f"and the shortening third doesn't even out-earn the bloating one. Across horizons the "
            f"long-short *flips sign* ({R['drift'][21][3]:+.2f}% at 1 month, {R['drift'][63][3]:+.2f}% "
            f"at 1 quarter, {R['drift'][126][3]:+.2f}% at 2 quarters). When the sign of your 'edge' "
            "wanders with the horizon and the era, you don't have an edge — you have noise. Compare "
            "the *margin* nudge above, which at least pointed one consistent way. Same signal: a "
            "faint truth about the business, nothing about the stock."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Precedes a better margin? — Faint.** Shortening-CCC names widen next quarter's "
            f"gross margin +{R['im_spread']*100:.2f} points more than bloating names, correctly "
            f"signed — but correlation just {R['im_corr']:+.2f}. Real in direction, negligible in "
            "size.\n"
            f"- **Signal (returns) — None.** The long-short is *wrong-signed* and statistically "
            f"zero (robust *t* = {R['ls_t_nw']:+.2f}), the thirds don't form a ladder (and the sign "
            f"flips across horizons and eras), and the conservative variants are flat. No "
            "certifiable return predictability on this tape.\n"
            "- **Tradability — Mirage.** It leans *negative* before costs and is deeply negative "
            "after them. Nothing to trade.\n\n"
            "> The honest one-liner: *on big liquid names, shortening the cash cycle is a "
            "real-but-tiny hint about next quarter's margin and — as far as this tape can tell — no "
            "usable hint at all about the stock.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where an edge might still hide.** Not in the *level* or naive *change* of a "
            "public ratio, but maybe in the *surprise* — CCC change vs a seasonal expectation — "
            "or in *which leg* moves (a payables stretch that shortens CCC is very different from "
            "clearing inventory into a demand slump). That decomposition is the residual this "
            "study didn't chase.\n"
            "- **The coverage caveat is real.** The CCC needs five matched facts a quarter; early "
            "in the sample the clean cross-section is thin, and terciles on a thin cross-section "
            "are noisy by construction.\n"
            "- **Sibling studies:** [853-days-sales-outstanding](../../853-days-sales-outstanding/) "
            "(the receivables leg alone), [529-inventory-growth](../../529-inventory-growth/) (the "
            "inventory leg), [153-net-operating-assets](../../153-net-operating-assets/) "
            "(balance-sheet bloat scaled by assets), and "
            "[524-operating-leverage](../../524-operating-leverage/) (cost structure). See "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the alpha is in the surprise or the leg mix, not the headline change? Build it, "
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
            "# Cash Conversion Cycle — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a pooled "
            "event-drift cross-check with a label-shuffle placebo · an era split · the "
            "precedes-margin regression · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **shortening the CCC frees cash, out-earns, and the stock follows** — splits "
            "into a mechanism piece and a return piece. This is distinct from every sibling on the "
            "desk: [853](../../853-days-sales-outstanding/) is the *receivables* leg alone, "
            "[529](../../529-inventory-growth/) the *inventory* leg, "
            "[153](../../153-net-operating-assets/) the balance-sheet-bloat *level* scaled by "
            "assets. This ranks on the **whole netted cycle**, DSO + DIO − DPO, year-over-year.\n\n"
            "> ⚠️ **Data note.** EDGAR `AccountsReceivableNetCurrent` + `InventoryNet` + "
            "`AccountsPayableCurrent` + `Revenues` + `CostOfRevenue` (COGS fallback "
            "`CostOfGoodsAndServicesSold`), longest per-name history, + yfinance adjusted closes. "
            "As-of 2026-06-30 — 30 names, 689 filing-quarters, period ends 2009-07-31 → "
            "2026-05-02. Point-in-time on the **filing date**. Survivorship named on the "
            "Signal axis (current-survivors basket). Thin/uneven five-fact coverage is a "
            "first-class caveat. Numbers in [`docs/results.md`](../docs/results.md) (prices "
            "fingerprint `d82e55a0af11`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** (returns) | `NONE` | calendar tercile long-short (long shortening / "
            f"short bloating) **{R['ls_mean_bps']:+.1f} bps/mo** ({R['ls_ann']:+.1f}%/yr gross), "
            f"one-sample *t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}** — "
            f"wrong-signed and insignificant; staleness-120 *t* = {R['ls120_t_nw']:+.2f}, "
            f"pct-change *t* = {R['pct_t_nw']:+.2f}; event drift flat and sign-flipping; sign flips "
            f"by era (+{R['era_early_t']:.2f} pre-2018, {R['era_late_t']:+.2f} post-2018) |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: {R['net'][(20, 100)][1]:+.2f}%/yr, "
            f"NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; **deeply "
            "negative** |\n"
            f"| **Precedes margin?** | `FAINT` | next-q gross-margin regression slope {R['im_slope']:+.5f} "
            f"(corr {R['im_corr']:+.2f}), shortening−bloating tercile spread **+{R['im_spread']*100:.2f} pp** — "
            "correctly signed, economically negligible |\n\n"
            "> 💡 In plain words: the working-capital 'virtue' is a wrong-signed null on returns "
            "and only a whisper on margin. On big, liquid, heavily-covered names, five accounting "
            "lines anyone can pull predict neither the stock nor (much) the profitability they're "
            "built from."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let name $i$'s cash conversion cycle at fiscal quarter $q$ be "
            "$\\text{CCC}_{i,q} = \\text{DSO}_{i,q} + \\text{DIO}_{i,q} - \\text{DPO}_{i,q}$, each "
            "leg a balance ÷ an annualised flow (receivables ÷ sales, inventory ÷ COGS, payables "
            "÷ COGS), disclosed on filing date $F_{i,q}$. The signal is the point-in-time YoY "
            "change $\\Delta_{i,q} = \\text{CCC}_{i,q} - \\text{CCC}_{i,q-4}$, known at $F_{i,q}$. "
            "The claims:\n\n"
            "- **H₁ (precedes margin).** A falling $\\Delta$ precedes a *widening* gross margin "
            "next quarter — the operating mechanism.\n"
            "- **H₂ (leads returns).** A cross-sectional long-short on $-\\Delta$ (long "
            "shorteners) earns a positive forward return spread — the mispricing claim.\n"
            "- **H₃ (tradable).** That spread survives realistic long-short costs + borrow.\n\n"
            f"We find **H₁ only faintly supported** (slope {R['im_slope']:+.5f}, corr "
            f"{R['im_corr']:+.2f}, +{R['im_spread']*100:.2f} pp tercile spread — right sign, trivial "
            f"size), **H₂ not supported and if anything reversed** (NW *t* = {R['ls_t_nw']:+.2f}; the "
            f"falling-CCC leg *under*-earns the rising-CCC leg; flat non-monotone event drift; the "
            f"sign flips across the 2018 break), and therefore **H₃ moot** (a deeply negative net "
            "spread). The operations-finance literature ties a shorter CCC to higher *profitability* "
            "in a broad cross-section; our large-cap, liquid tape is the hardest place for that to "
            "translate into a *return* edge, and it does not — hence a `NONE`, not a `WEAK`, on "
            "returns."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, precisely "
            "because balance-sheet signals are persistent and filings cluster: a calendar series "
            "of monthly long-short returns lets a **Newey-West (6-lag) HAC *t*** do the honest "
            "work the desk's `REAL` bar is written against. The panel is thin (five matched facts "
            "per quarter), so we sort into **terciles** (not quintiles) and require ≥ 6 names in "
            "the cross-section. The pooled event drift + a **label-shuffle placebo** is the "
            "cross-check; the precedes-margin regression is graded on **magnitude** (its pooled "
            "*t* ignores quarter clustering)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) CCC quarters across {R['n_names']} "
            f"names, period ends {R['end_lo']} → {R['end_hi']}, each stamped with its 10-Q/10-K "
            f"filing date (point-in-time). The CCC needs five matched facts a quarter, so the clean "
            f"cross-section (avg {R['avg_n']:.1f} names/month) is thinner than the 45-name price "
            "basket.\n"
            "- **Primary.** Monthly tercile long-short on `ccc_score` (= −ΔCCC), one execution lag "
            "(rank at month $M$ close, earn month $M{+}1$); Newey-West + one-sample *t*, Sharpe, "
            "Wilson-interval hit rate.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag entry, "
            "top-minus-bottom tercile, one-sample *t* + 10k-draw label-shuffle placebo, and the "
            "tercile monotonicity picture.\n"
            "- **Robustness.** Staleness 120 vs 200 days; the unit-free percentage-change signal "
            "`ΔCCC/CCC`; an era split at 2018 (the ASC-606 revenue-tagging break).\n"
            "- **Mechanism.** Pooled OLS of next-quarter gross-margin change on `ccc_yoy_chg`; "
            "expected slope < 0.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short "
            "borrow.\n"
            "- **Control.** Synthetic panel, planted-lead knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh signals into terciles each month, long the shortening third / short the "
            "bloating third equal-weight, earn next month's return. The decisive statistic is the "
            "HAC *t* of the monthly long-short series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='ccc_score', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls_pct = st.calendar_ls(PX, EV, signal_col='ccc_pct_score', n_buckets=3, min_names=6, staleness_days=200)\n"
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
            "    a2.set_title('Thin & uneven: the five-fact panel')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-120 NW t = {R['ls120_t_nw']:+.2f}, pct-change NW t = {R['pct_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **{R['ls_mean_bps']:+.0f} bps/month** ({R['ls_ann']:+.1f}%/yr "
            f"gross) is *wrong-signed* — the falling-CCC (long) leg under-earns the rising-CCC "
            f"(short) leg, the opposite of the efficiency claim — and the HAC *t* is "
            f"**{R['ls_t_nw']:+.2f}**, indistinguishable from zero over n={R['n_months']} months "
            f"(avg cross-section {R['avg_n']:.0f}). The point estimate isn't even stably negative: "
            f"the staleness-120 cut is {R['ls120_mean_bps']:+.0f} bps (*t* = {R['ls120_t_nw']:+.2f}) "
            f"and the unit-free percentage-change signal is a flat {R['pct_mean_bps']:+.0f} bps "
            f"(*t* = {R['pct_t_nw']:+.2f}). Whatever this is, it isn't a robust sort in the claimed "
            "direction — and the full-sample negative is itself a post-2018 artefact (see 4c)."
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
            "a1.set_ylabel('one-sample t (long-short drift)'); a1.set_title('Every horizon vs |t|=2')\n"
            "a2.bar(['bloating','middle','shortening'], mono, color=GREY, width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('Tercile ladder (bloating->shortening)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift is **flat** and *changes sign* with "
            f"the horizon — *t* = {R['drift'][21][4]:+.2f} at 21d, {R['drift'][63][4]:+.2f} at 63d, "
            f"{R['drift'][126][4]:+.2f} at 126d — never within shouting distance of ±2. The "
            f"label-shuffle placebo *p* runs **{R['drift'][21][6]:.2f}-{R['drift'][63][6]:.2f}** (a "
            f"random tercile split matches or beats the real one about half the time), and the 63d "
            f"terciles are **non-monotone** ({R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / "
            f"{R['mono63'][2]:+.2f}% bloating→shortening — a dip in the middle, and the shortening "
            "third doesn't top the bloating one). A horizon-dependent sign on a non-monotone ladder "
            "is the signature of noise, not a signal."
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
            "ax.set_title('Calendar long-short by era')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2010-2017: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2018-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: the sign **flips across the 2018 break** — mildly positive before "
            f"(+{R['era_early_bps']:.0f} bps/mo, NW *t* = +{R['era_early_t']:.2f}, short of the bar) "
            f"and firmly negative after ({R['era_late_bps']:+.0f} bps/mo, NW *t* = "
            f"{R['era_late_t']:+.2f}). The full-sample negative is really a post-2018 artefact, not "
            f"a stable effect, and neither half certifies. Note the clean cross-section actually "
            f"*shrinks* (≈{R['xsec_early']:.0f} matched names early, ≈{R['xsec_late']:.0f} late) as "
            "ASC-606 revenue-tag switches break the same-fiscal-quarter YoY match — so the modern "
            "half is both thinner and the one dragging the sign negative. There is no era in which "
            "this certifies in the claimed direction."
        ),
        md(
            "### 4d · The mechanism — does shortening the CCC precede a better margin?\n\n"
            "Pooled OLS of next-quarter gross-margin change on this quarter's CCC change, and the "
            "future-margin spread between the shortening and bloating terciles."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.improves_margin(EV)\n"
            "    fr = EV.dropna(subset=['ccc_yoy_chg','next_gm_chg'])\n"
            "    x, y = fr['ccc_yoy_chg'].to_numpy(), fr['next_gm_chg'].to_numpy()\n"
            "    slope, corr, lo, hi = q['slope'], q['corr'], q['low_margin']*100, q['high_margin']*100\n"
            "else:\n"
            "    x = y = None\n"
            "    slope, corr, lo, hi = R['im_slope'], R['im_corr'], R['im_low']*100, R['im_high']*100\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if x is not None:\n"
            "    m = (np.abs(x)<np.percentile(np.abs(x),98))&(np.abs(y)<np.percentile(np.abs(y),98))\n"
            "    a1.scatter(x[m], y[m]*100, s=8, alpha=.25, color=GREEN)\n"
            "    xs = np.linspace(np.percentile(x,2), np.percentile(x,98), 50)\n"
            "    a1.plot(xs, (q['slope']*xs + (y.mean()-q['slope']*x.mean()))*100, color=RED, lw=2)\n"
            "    a1.set_xlabel('CCC change (days)'); a1.set_ylabel('NEXT-quarter gross-margin change (pp)')\n"
            "    a1.set_title(f'Precedes margin: slope {slope:+.4f}, corr {corr:+.2f}')\n"
            "else:\n"
            "    a1.text(.5,.5,'run with cache',ha='center'); a1.set_axis_off()\n"
            "a2.bar(['bloating third','shortening third'], [hi, lo], color=[GREY, GREEN], width=.5)\n"
            "for i,v in enumerate([hi, lo]): a2.annotate(f'{v:+.2f}pp',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel(\"next quarter's margin change\"); a2.set_title(f'{lo-hi:+.2f} pp future-margin spread')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'precedes-margin: slope {slope:+.5f}, corr {corr:+.3f}, future-margin spread {lo-hi:+.2f} pp (shortening {lo:+.2f}pp vs bloating {hi:+.2f}pp)')"
        ),
        md(
            f"> 💡 In plain words: this is the study's *strongest* result and it's still faint. The "
            f"slope is **{R['im_slope']:+.5f}** (correctly negative — more CCC shortening, wider "
            f"next-quarter margin) and the shortening third widens margin **+{R['im_spread']*100:.2f} "
            f"pp** more than the bloating third — but the correlation is **{R['im_corr']:+.2f}**, R² "
            f"**{R['im_r2']:.3f}**, and the pooled *t* is only **{R['im_t']:+.2f}** (and even that "
            "ignores quarter clustering). We read the magnitude, and the magnitude says the 'frees "
            "cash → out-earns' mechanism is a real-but-negligible tilt on these names, not a lever."
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
            f"> 💡 In plain words: turnover is modest (~{R['ls_turn']:.2f}/mo), but there is no gross "
            f"edge to protect — it was negative to begin with. Light friction (10 bps + 50 bps "
            f"borrow) leaves {R['net'][(10, 50)][1]:+.2f}%/yr (*t* = {R['net'][(10, 50)][2]:+.2f}); a "
            f"realistic 20 bps + 100 bps borrow deepens it to {R['net'][(20, 100)][1]:+.2f}%/yr "
            f"(*t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}). "
            "**Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + signal panel with a TUNABLE planted lead (shortening-CCC names "
            "drift up). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=854 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=854)\n"
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
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 in "
            f"**{R['syn_null_fire']}/12** seeds — an unbiased, well-behaved null. A planted lead "
            f"reads NW *t* = **{R['syn_planted_t']:.1f}** ({R['syn_planted_bps']:.0f} bps/mo). So the "
            f"machinery is faithful and powerful, which means the real-tape {R['ls_t_nw']:+.2f} is a "
            "genuine null (indeed a wrong-signed one), not a broken pipeline. *(Power check only — "
            "never cited in support of a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `NONE`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo ({R['ls_ann']:+.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**, and *wrong-signed* (the falling-CCC leg "
            f"under-earns the rising-CCC leg); staleness-120 and percentage-change variants also "
            f"fail ({R['ls120_t_nw']:+.2f}, {R['pct_t_nw']:+.2f}); pooled event drift flat and "
            f"sign-flipping (placebo *p* {R['drift'][21][6]:.2f}-{R['drift'][63][6]:.2f}); the sign "
            f"flips across the 2018 break (+{R['era_early_t']:.2f} → {R['era_late_t']:+.2f}). No "
            "coherent, robust effect in the claimed direction — the desk's `NONE`.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: {R['net'][(20, 100)][1]:+.2f}%/yr, "
            f"NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; deeply "
            "negative.\n"
            f"- **Precedes margin? `FAINT`** — regression slope {R['im_slope']:+.5f}, correlation "
            f"{R['im_corr']:+.2f}, **+{R['im_spread']*100:.2f} pp** shortening-minus-bloating "
            "future-margin spread. Correctly signed, economically negligible — the mechanism exists "
            "but is a whisper, not the lever the efficiency story promises."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The residual worth chasing** is the *surprise* and the *leg mix*, not the "
            "headline change: a CCC that falls because payables were stretched (a liquidity "
            "warning) is nothing like one that falls because collections improved. Decomposing "
            "$\\Delta$CCC into its DSO/DIO/DPO contributions, or benchmarking against a seasonal "
            "expectation, is where any hidden edge would live.\n"
            "- **Coverage honesty:** the five-fact panel is thin early; terciles on a thin "
            "cross-section are noisy, and the pooled event drift across all events is the more "
            "decisive evidence.\n"
            "- **Dedup map:** [853-days-sales-outstanding](../../853-days-sales-outstanding/) "
            "(receivables leg), [529-inventory-growth](../../529-inventory-growth/) (inventory "
            "leg), [153-net-operating-assets](../../153-net-operating-assets/) (balance-sheet "
            "bloat level / assets), [524-operating-leverage](../../524-operating-leverage/) (cost "
            "structure). None ranks on the netted DSO + DIO − DPO cycle change itself.\n\n"
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
