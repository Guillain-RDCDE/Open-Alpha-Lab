"""Generate the two narrative notebooks for Study 860 (Accounting Conservatism, C-Score).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR reserve/allowance accounts
# + yfinance prices, 37 non-financial names, period ends 2008-09 -> 2026-05, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=37, n_events=1669,
    end_lo="2008-09-27", end_hi="2026-05-31", fp_prices="5506b659db08",
    cscore_med=0.34, cscore_mean=0.98,
    # primary calendar long-short (cscore, terciles, staleness 200)
    n_months=202, avg_n=28.5, ls_span_lo="2009-09-30", ls_span_hi="2026-06-30",
    ls_mean_bps=4.0, ls_ann=0.49, ls_t_iid=0.21, ls_t_nw=0.22, ls_sharpe=0.05,
    ls_hit=49, ls_long_bps=152.4, ls_short_bps=148.4, ls_turn=0.21, ls_cum=1.005,
    ls120_mean_bps=-9.4, ls120_t_nw=-0.43,
    noa_n=194, noa_mean_bps=0.6, noa_t_nw=0.02,
    xsec_early=20, xsec_late=31, noa_cov=42,
    # pooled event drift  horizon -> (n, top%, bot%, ls%, t, win%, placebo p)
    drift={
        21: (1667, 1.40, 1.47, -0.07, -0.17, 49, 0.5650),
        63: (1646, 3.57, 4.26, -0.68, -0.91, 49, 0.8345),
        126: (1615, 6.55, 7.91, -1.36, -1.31, 48, 0.9211),
    },
    mono63=(4.26, 2.57, 3.57), mono126=(7.91, 5.62, 6.55),
    # era split (calendar LS, split 2016)
    era_early_n=76, era_early_bps=17.8, era_early_t=0.55,
    era_late_n=126, era_late_bps=-4.3, era_late_t=-0.20,
    # Basu asymmetric timeliness
    basu_n=1607, basu_b_good=0.006, basu_b_asym=0.025, basu_t=1.67, basu_r2=0.005,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (-4.3, -0.51, -0.24, -0.05), (20, 100): (-12.5, -1.51, -0.70, -0.16)},
    # synthetic control
    syn_null_mean=0.47, syn_null_sd=0.76, syn_null_fire=0, syn_null_seeds=12,
    syn_planted_bps=1057.5, syn_planted_t=26.57,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Conservative%3F: Suggestive](https://img.shields.io/badge/Conservative%3F-Suggestive-8b949e?style=flat-square)\n\n"
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

from conservatism import data, strategy as st

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
            "# Hidden reserves on the balance sheet. Buried treasure, or just buried? 🛡️\n"
            "### A conservatism score with real accounting theory behind it — and a stock "
            "return of exactly nothing\n\n"
            + BADGES +
            "Here's a signal with a Nobel-adjacent pedigree. *Conservative* accounting books bad "
            "news the instant it's plausible but makes good news wait until it's certain — the "
            "prudent, lower-of-cost-or-market instinct. A side effect is that conservative firms "
            "quietly stuff the balance sheet with **hidden reserves**: allowances for bad debts, "
            "inventory write-down reserves, valuation allowances. Book value understates economic "
            "value, and those reserves later melt back into earnings. The pitch writes itself: "
            "find the most conservative firms — the ones sitting on the biggest reserve cushion — "
            "and you've found un-booked value the market hasn't paid for yet. Buy them.\n\n"
            "It's a lovely story. The accounting part is even mildly true. The *buy them* part is "
            "where it falls apart.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the Basu regression and "
            "the cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 37 US non-financial names that tag reserve/allowance accounts "
            "on EDGAR, 2008→2026; a genuinely **coarse proxy** (XBRL exposes only a slice of the "
            "reserves the theory wants) on a **thin, uneven panel**. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the accounting actually **conservative**? | **Mildly.** Earnings react a little "
            f"more to bad news than good (Basu asymmetry +{R['basu_b_asym']:.3f}), but faintly "
            f"(*t* = +{R['basu_t']:.2f}, explaining ~0.5% of the variation). A real property, "
            "weakly present. |\n"
            f"| Does a high conservatism score predict future **returns**? | **No.** A long-short "
            f"that buys the most-conservative names and shorts the least earns "
            f"**+{R['ls_ann']:.1f}%/yr** gross — a flat line, \\$1 → \\${R['ls_cum']:.3f} over 17 "
            f"years (robust *t* = **+{R['ls_t_nw']:.2f}**, the bar is 2). If anything the "
            "most-conservative names *lag*. |\n"
            "| Why the gap? | The reserve balances are **public** and the score is **coarse**. "
            "Anyone can read the allowance line; the market prices whatever it means on filing "
            "day. A faint accounting property is not a stock signal. |\n"
            "| Can you trade it? | **No.** There's no gross edge to charge costs against — net of "
            "friction the book loses money. |\n\n"
            "> A real (if weak) accounting phenomenon. No alpha in the stock. That distinction is "
            "the whole study."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Conservative accounting hides value. Prudent firms build reserves that "
            "understate their true worth, and those reserves unwind into earnings later — so buy "
            "the firms carrying the biggest reserve cushion.\"*\n\n"
            "It's the strong form of a respectable academic idea. **Basu (1997)** defined "
            "conservatism as earnings reacting faster to bad news than good; **Penman & Zhang "
            "(2002)** built a *C-score* — estimated reserves over net operating assets — and "
            "argued it signals earnings quality and future returns. The question is whether that "
            "accounting property is also a *trading* signal."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a plain balance-sheet ratio — reserves divided by assets — predicted stock "
            "returns, it would be one of the easiest anomalies going: no estimates feed, no "
            "alt-data, just a couple of lines from every 10-Q. That's exactly why it deserves "
            "suspicion. Everyone with a brokerage account can pull the allowance for doubtful "
            "accounts. If markets price public information at all, this is the kind they should "
            "price fastest."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The score.** For each of {R['n_names']} non-financial names, add up the tagged "
            "reserve accounts (bad-debt allowance + inventory reserve + deferred-tax valuation "
            "allowance) and divide by assets — known only on the **filing date** of the "
            "10-Q/10-K (no peeking).\n"
            "- **The conservatism test.** Does the panel actually book bad news faster than good "
            "(the Basu asymmetry)? If not, the score can't even be a proxy for what it claims.\n"
            "- **The return test.** Each month, rank the names on the score, buy the most "
            "conservative third, short the least, hold for the next month. Does the spread make "
            "money — and can we tell it apart from noise?\n"
            "- **The mirage check.** If the return spread can't beat a coin-flip relabelling of "
            "the names, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The money question first: do the most-conservative stocks out-return the least?** "
            "Rank on the score each month, buy the top third, short the bottom third, and track "
            "the growth of \\$1."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='cscore', n_buckets=3, min_names=6, staleness_days=200)\n"
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
            "    ax.set_title(f'A line going nowhere: +{ann:.1f}%/yr but robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: +{ann:.1f}%/yr gross, Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"That is about as flat as a strategy line gets: **+{R['ls_ann']:.1f}%/yr** gross, "
            f"\\$1 growing to \\${R['ls_cum']:.3f} over {R['n_months']} months, robust *t* = "
            f"**+{R['ls_t_nw']:.2f}** against a bar of 2. Not a decaying edge, not a costed-away "
            "edge — just **no edge**.\n\n"
            "**The tell:** if the score really sorted returns, the middle third would land between "
            "the top and bottom, and the top (most conservative) would win. Instead the thirds "
            "are jumbled — and if anything the *most*-conservative names trail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=126)\n"
            "    mono = st.bucket_means(fr, 3)*100\n"
            "else:\n"
            "    mono = np.array(R['mono126'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['bottom third\\n(least\\nconservative)','middle\\nthird','top third\\n(most\\nconservative)'], mono,\n"
            "       color=[GREY, GREY, RED], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 6-month forward return')\n"
            "ax.set_title('No ladder — and the most-conservative third actually trails')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('6-month forward return by conservatism third (low->high):', [f'{v:+.2f}%' for v in mono])"
        ),
        md(
            f"No staircase — and the wrong slope. Over the next two quarters the *least*-"
            f"conservative third returns **+{R['mono126'][0]:.1f}%** and the *most*-conservative "
            f"third only **+{R['mono126'][2]:.1f}%**. A random relabelling of the names beats the "
            "real long-short ~9 times in 10. Whatever hidden reserves are doing for the business, "
            "they are not buying these stocks a higher return.\n\n"
            "**So is the accounting even conservative in the first place?** We check the Basu "
            "signature directly: do earnings react more to bad news (negative stock returns) than "
            "to good?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.basu_asymmetry(EV)\n"
            "    b_good, b_asym, t_asym = q['b_good'], q['b_asym'], q['t_asym']\n"
            "else:\n"
            "    b_good, b_asym, t_asym = R['basu_b_good'], R['basu_b_asym'], R['basu_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar(['good news\\n(earnings sensitivity)','bad news\\n(extra sensitivity)'],\n"
            "       [b_good, b_asym], color=[GREY, AMBER], width=.5)\n"
            "for i,v in enumerate([b_good, b_asym]): ax.annotate(f'{v:+.3f}',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('Basu slope (ROA on return)')\n"
            "ax.set_title(f'Mildly conservative: bad news bites a bit harder (t={t_asym:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Basu: good-news slope {b_good:+.3f}, extra bad-news slope {b_asym:+.3f} (t={t_asym:+.2f})')"
        ),
        md(
            f"There *is* a bad-news asymmetry — the extra sensitivity to bad news is "
            f"**+{R['basu_b_asym']:.3f}** — so the panel is genuinely, if faintly, conservative "
            f"(*t* = +{R['basu_t']:.2f}; strong-but-not-overwhelming). The accounting theory "
            "isn't fantasy. It just doesn't convert into a stock edge: the conservatism is real, "
            "weak, public, and already priced."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) — None.** The long-short is a flat +{R['ls_ann']:.1f}%/yr with "
            f"robust *t* = +{R['ls_t_nw']:.2f}, non-monotone thirds, a placebo the real strategy "
            "*loses* to, and (if anything) the most-conservative names trailing. No premium here.\n"
            "- **Tradability — Mirage.** No gross edge to monetise; net of costs the book loses "
            "money.\n"
            "- **Is it conservative? — Suggestive.** A faint but real Basu bad-news asymmetry "
            f"(+{R['basu_b_asym']:.3f}, *t* = +{R['basu_t']:.2f}). The property exists; the coarse "
            "reserve proxy captures it poorly; and it buys no return.\n\n"
            "> The honest one-liner: *conservative firms really do hide a little value — and the "
            "market has already counted it.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where an edge might still hide.** Not in the *level* of a public reserve ratio, "
            "but maybe in the *change* — reserve *releases* that flatter earnings (the Penman-"
            "Zhang change-in-conservatism story), or a sharper C-score built from the reserves "
            "XBRL doesn't expose (LIFO, capitalised R&D and advertising). That's the residual "
            "this coarse study didn't chase.\n"
            "- **The coverage caveat is real.** Reserve tags are irregular — most names disclose "
            "only the bad-debt allowance, and only in some years — so our score is a *floor* on "
            f"true reserves and the cross-section is thin (≈{R['xsec_early']} names early, "
            f"≈{R['xsec_late']} late; only {R['noa_cov']}% carry the cleaner NOA denominator).\n"
            "- **Sibling studies:** [229-beneish-m-score](../../229-beneish-m-score/) (detects the "
            "*opposite* — income-inflating manipulation), [232-mohanram-g-score](../../232-mohanram-g-score/) "
            "(a growth-firm fundamental composite), [855-accrual-quality](../../855-accrual-quality/) "
            "(how well accruals map to cash), and [52-smoke-screen](../../52-smoke-screen/) (the "
            "method-demo cousin of a good-story-no-reward null). See "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the alpha is in reserve *changes*, not levels? Build the release signal, show "
            "a certifiable net spread on the size you'd actually run — then we'll talk.*"
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
            "# Accounting Conservatism (C-Score) — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a pooled "
            "event-drift cross-check with a label-shuffle placebo · an era split · a NOA-scaled "
            "variant · the Basu asymmetric-timeliness regression · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — a **Penman-Zhang / Basu conservatism score leads returns** — splits into two "
            "testable pieces: *is the accounting conservative* (Basu) and *does the score sort "
            "returns* (the long-short). The first is faintly true; the second is a clean null. "
            "This is distinct from every sibling on the desk: [229](../../229-beneish-m-score/) "
            "detects income-*inflating* manipulation, [232](../../232-mohanram-g-score/) is a "
            "growth composite, [855](../../855-accrual-quality/) measures accrual-to-cash "
            "mapping.\n\n"
            "> ⚠️ **Data note.** EDGAR reserve/allowance tags (`AllowanceForDoubtfulAccounts"
            "ReceivableCurrent` + `InventoryValuationReserves` + `DeferredTaxAssetsValuation"
            "Allowance`) ÷ Assets + yfinance adjusted closes, "
            + R["n_names"].__str__() + " names, ends "
            + R["end_lo"] + " → " + R["end_hi"] + ", as-of " + R["as_of"] + ". Point-in-time on "
            "the **filing date**. Survivorship named on the Signal axis (current-survivors "
            "basket). A **coarse reserve floor** on thin/uneven coverage is a first-class caveat. "
            "Numbers in [`docs/results.md`](../docs/results.md) (prices fingerprint `"
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
            f"**{R['ls_mean_bps']:+.1f} bps/mo** (+{R['ls_ann']:.1f}%/yr gross), one-sample "
            f"*t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; staleness-120 "
            f"flips to {R['ls120_mean_bps']:+.1f} bps (NW *t* = {R['ls120_t_nw']:+.2f}), "
            f"NOA-scaled NW *t* = {R['noa_t_nw']:+.2f}; pooled drift negative, sign flips by era |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: "
            f"{R['net'][(20, 100)][1]:+.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20, 100)][3]:.2f}; no gross edge to begin with |\n"
            f"| **Is the accounting conservative?** | `SUGGESTIVE` | Basu asymmetry "
            f"$b_3$ = +{R['basu_b_asym']:.3f} (*t* = +{R['basu_t']:.2f}, R² = {R['basu_r2']:.3f}) "
            "— faintly positive |\n\n"
            "> 💡 In plain words: the panel is *mildly* conservative and the return edge is a "
            "flat (if anything wrong-signed) null. A coarse public reserve ratio proxies a "
            "faint accounting property that the market has already priced."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{Res}_{i,q}$ be name $i$'s tagged estimated reserves at fiscal quarter "
            "$q$, disclosed on filing date $F_{i,q}$. The C-score is the point-in-time reserve "
            "intensity $c_{i,q} = \\text{Res}_{i,q}/\\text{Assets}_{i,q}$, known at $F_{i,q}$. The "
            "claims:\n\n"
            "- **H₁ (conservative accounting).** The panel books bad news faster than good — a "
            "positive Basu interaction $b_3$ in $\\text{ROA} = a + b_1 R + b_2 D + b_3 (D\\cdot R)$, "
            "$D = \\mathbb{1}[R<0]$.\n"
            "- **H₂ (leads returns).** A cross-sectional long-short on $c$ (long high / short low) "
            "earns a positive forward return spread — the market-mispricing claim.\n"
            "- **H₃ (tradable).** That spread survives realistic long-short costs + borrow.\n\n"
            f"We find **H₁ weakly supported** ($b_3$ = +{R['basu_b_asym']:.3f}, *t* = "
            f"+{R['basu_t']:.2f}), **H₂ not supported** (NW *t* = {R['ls_t_nw']:+.2f}, negative "
            "and non-monotone event drift), and therefore **H₃ moot** (nothing to trade). The "
            "literature (Basu 1997; Penman-Zhang 2002) supports a real conservatism property; our "
            "coarse tape captures it faintly and finds **no** associated return premium — hence a "
            "`NONE` on returns, not a `WEAK`: the point estimate is essentially zero and not even "
            "robustly right-signed."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, precisely "
            "because balance-sheet signals are persistent and filings cluster: a calendar series "
            "of monthly long-short returns lets a **Newey-West (6-lag) HAC *t*** do the honest "
            "work the desk's `REAL` bar is written against. The panel is thin, so we sort into "
            "**terciles** (not quintiles) and require ≥ 6 names in the cross-section. The pooled "
            "event drift + a **label-shuffle placebo** is the cross-check; the Basu regression is "
            "graded on **sign and magnitude** (its pooled *t* ignores quarter clustering)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) reserve-disclosing quarters across "
            f"{R['n_names']} names, ends {R['end_lo']} → {R['end_hi']}, each stamped with its "
            "10-Q/10-K filing date (point-in-time).\n"
            "- **Primary.** Monthly tercile long-short on `cscore`, one execution lag (rank at "
            "month $M$ close, earn month $M{+}1$); Newey-West + one-sample *t*, Sharpe, hit rate.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag entry, "
            "top-minus-bottom tercile, one-sample *t* + 10k-draw label-shuffle placebo, and the "
            "tercile monotonicity picture.\n"
            "- **Robustness.** Staleness 120 vs 200 days; the NOA-scaled signal "
            "`reserves/NOA`; an era split at 2016.\n"
            "- **Mechanism.** Pooled Basu (1997) asymmetric-timeliness regression.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short "
            "borrow.\n"
            "- **Control.** Synthetic panel, planted-relation knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh scores into terciles each month, long the most conservative / short the "
            "least equal-weight, earn next month's return. The decisive statistic is the HAC *t* "
            "of the monthly long-short series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='cscore', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls_noa = st.calendar_ls(PX, EV, signal_col='cscore_noa', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s_noa = st.calendar_ls_stats(ls_noa)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo (+{s['ann_pct']:.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  NOA-scaled signal: {s_noa['mean_bps']:+.1f} bps/mo, NW t = {s_noa['t_nw']:+.2f}, n={s_noa['n_months']}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=RED, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: +{R['ls_ann']:.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven coverage')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-120 NW t = {R['ls120_t_nw']:+.2f}, NOA-scaled NW t = {R['noa_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **+{R['ls_mean_bps']:.0f} bps/month** is a rounding error "
            f"(~+{R['ls_ann']:.1f}%/yr gross), and the HAC *t* is **+{R['ls_t_nw']:.2f}** — a flat "
            f"null over n={R['n_months']} months. Every specification agrees and none is even "
            f"reliably positive: staleness-120 NW *t* = {R['ls120_t_nw']:+.2f} (negative), the "
            f"cleaner NOA-scaled signal NW *t* = {R['noa_t_nw']:+.2f} (dead zero). There is no "
            "conservatism return premium to certify — right-signed or otherwise."
        ),
        md(
            "### 4b · The cross-check — pooled event drift + placebo + monotonicity\n\n"
            "Bucket all events by the score; top-minus-bottom forward drift with a label-shuffle "
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
            "a1.set_ylabel('one-sample t (long-short drift)'); a1.set_title('Flat / negative: no horizon clears |t|=2')\n"
            "a2.bar(['bottom','middle','top'], mono, color=[GREY, GREY, RED], width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('No ladder (non-monotone terciles)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift is **negative** at every horizon "
            f"(*t* from {R['drift'][21][4]:+.2f} to {R['drift'][126][4]:+.2f}), the label-shuffle "
            f"placebo *p* climbs to **{R['drift'][126][6]:.2f}** (a random tercile split beats the "
            f"real one ~9 times in 10, because the real one is mildly wrong-signed), and the "
            f"terciles are **non-monotone** ({R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / "
            f"{R['mono63'][2]:+.2f}% low→high at 63d). The event study and the calendar long-short "
            "agree: no return sort in the claimed direction."
        ),
        md(
            "### 4c · Era split — nothing hiding in a regime\n\n"
            "Split the calendar long-short at 2016."
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
            "ax.bar([f'2009-2015\\n(n={en})', f'2016-2026\\n(n={ln})'], [eb, lb], color=[GREY, GREY], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Sign flips across eras; neither significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2009-2015: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2016-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: {R['era_early_bps']:+.0f} bps (NW *t* = {R['era_early_t']:+.2f}) "
            f"early, {R['era_late_bps']:+.0f} bps (NW *t* = {R['era_late_t']:+.2f}) late — the "
            "**sign flips** and neither half certifies. Not a live edge that decayed; it never "
            "existed."
        ),
        md(
            "### 4d · The mechanism — is the panel actually conservative? (Basu 1997)\n\n"
            "Pooled regression of quarterly ROA on the contemporaneous return, a bad-news dummy, "
            "and their interaction. The interaction slope $b_3$ is Basu's asymmetric timeliness."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.basu_asymmetry(EV)\n"
            "    fr = EV.dropna(subset=['roa','ret_contemp'])\n"
            "    r, y = fr['ret_contemp'].to_numpy(), fr['roa'].to_numpy()\n"
            "    m = (np.abs(r)<1.5)&(np.abs(y)<0.5)\n"
            "    r, y = r[m], y[m]\n"
            "    b_good, b_asym, t_asym = q['b_good'], q['b_asym'], q['t_asym']\n"
            "else:\n"
            "    r = y = None\n"
            "    b_good, b_asym, t_asym = R['basu_b_good'], R['basu_b_asym'], R['basu_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if r is not None:\n"
            "    a1.scatter(r*100, y*100, s=6, alpha=.15, color=AMBER)\n"
            "    xs_neg = np.linspace(np.percentile(r,2), 0, 25); xs_pos = np.linspace(0, np.percentile(r,98), 25)\n"
            "    a = y.mean() - (b_good)*r.mean()\n"
            "    a1.plot(xs_pos*100, (a + b_good*xs_pos)*100, color=GREY, lw=2, label='good news')\n"
            "    a1.plot(xs_neg*100, (a + (b_good+b_asym)*xs_neg)*100, color=RED, lw=2, label='bad news (steeper)')\n"
            "    a1.set_xlabel('contemporaneous return (%)'); a1.set_ylabel('quarterly ROA (%)'); a1.legend()\n"
            "    a1.set_title(f'Basu: bad-news slope steeper (asym {b_asym:+.3f})')\n"
            "else:\n"
            "    a1.text(.5,.5,'run with cache',ha='center'); a1.set_axis_off()\n"
            "a2.bar(['good-news\\nslope $b_1$','bad-news\\nextra $b_3$'], [b_good, b_asym], color=[GREY, AMBER], width=.5)\n"
            "for i,v in enumerate([b_good, b_asym]): a2.annotate(f'{v:+.3f}',(i,v),ha='center',va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('Basu slope'); a2.set_title(f'Asymmetry t = {t_asym:+.2f} (faint)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Basu: good-news b1 {b_good:+.3f}, bad-news extra b3 {b_asym:+.3f}, t(b3) {t_asym:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the interaction is **positive** ($b_3$ = "
            f"+{R['basu_b_asym']:.3f}) — earnings do bite a little harder on bad news — so the "
            f"panel is **genuinely, faintly conservative** (*t* = +{R['basu_t']:.2f}, R² = "
            f"{R['basu_r2']:.3f}). The accounting property the C-score means to proxy is real but "
            "weak on this coarse quarterly measure. The return null is therefore **not** a "
            "machinery failure — it's a market-efficiency result: a faint, public accounting "
            "property carries no forward-return edge. (The Basu *t* is pooled across clustered "
            "filings; we read sign + magnitude, not the literal value.)"
        ),
        md(
            "### 4e · Tradability — the timer\n\n"
            "For completeness, the calendar long-short net of one-way costs × turnover (both "
            "legs) + short borrow — though a near-zero gross *t* already settles it."
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
            "ax.set_title('Net of friction the book loses money — and there was no gross edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: {a:+.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: turnover is modest (~{R['ls_turn']:.2f}/mo) but irrelevant — "
            f"the gross edge was already zero, so net of 20 bps + 100 bps borrow the book runs "
            f"**{R['net'][(20,100)][1]:+.1f}%/yr** (NW *t* = {R['net'][(20,100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20,100)][3]:.2f}). **Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + score panel with a TUNABLE planted relation (high-conservatism "
            "names drift up). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=860 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=860)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [min(planted_t, 12)], color=GREEN, s=95, zorder=5, label='planted relation (edge=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null never fires; a planted relation lights up hard'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 "
            f"{R['syn_null_fire']}/12 times; a planted relation reads NW *t* = "
            f"{R['syn_planted_t']:.1f}. The machinery is unbiased and powered, so the real-tape "
            f"+{R['ls_t_nw']:.2f} is a genuine null, not a broken pipeline. *(Power check only — "
            "never cited in support of a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `NONE`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo (+{R['ls_ann']:.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; staleness-120 {R['ls120_mean_bps']:+.1f} "
            f"bps (NW *t* = {R['ls120_t_nw']:+.2f}), NOA-scaled NW *t* = {R['noa_t_nw']:+.2f}; "
            "pooled event drift **negative** and non-monotone (placebo *p* up to "
            f"{R['drift'][126][6]:.2f}); the sign flips across eras. A flat, not-even-right-signed "
            "null.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: "
            f"{R['net'][(20, 100)][1]:+.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20, 100)][3]:.2f}; no gross edge to monetise.\n"
            f"- **Is the accounting conservative? `SUGGESTIVE`** — Basu asymmetry $b_3$ = "
            f"+{R['basu_b_asym']:.3f}, *t* = +{R['basu_t']:.2f}, R² = {R['basu_r2']:.3f}. A "
            "faint, real bad-news asymmetry; the property exists but is weak, coarsely proxied, "
            "and already priced."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The residual worth chasing** is the *change* in conservatism, not the *level*: "
            "reserve *releases* that flatter earnings (the Penman-Zhang temporal story), or a "
            "sharper C-score built from the reserves XBRL doesn't expose (LIFO, capitalised R&D, "
            "advertising). A faint public level that the market already prices is the textbook "
            "signature of an efficiently-priced fundamental.\n"
            "- **Coverage honesty:** reserve tags are irregular — most names disclose only the "
            f"bad-debt allowance, and only some years — so our score is a *floor* on true reserves "
            f"and the cross-section is thin (≈{R['xsec_early']} names early → ≈{R['xsec_late']} "
            f"late; only {R['noa_cov']}% carry the NOA denominator). The flat/negative event drift "
            "across 1,600+ pooled events is the more decisive evidence of the return null.\n"
            "- **Dedup map:** [229-beneish-m-score](../../229-beneish-m-score/) (detects the "
            "*opposite* — income-inflating manipulation), [232-mohanram-g-score](../../232-mohanram-g-score/) "
            "(a growth-firm fundamental composite), [855-accrual-quality](../../855-accrual-quality/) "
            "(how well accruals map to cash), [52-smoke-screen](../../52-smoke-screen/) (a "
            "method demo of spurious accounting overlays). None ranks on reserve-intensity "
            "conservatism itself.\n\n"
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
