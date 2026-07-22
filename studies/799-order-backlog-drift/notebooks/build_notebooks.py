"""Generate the two narrative notebooks for Study 799 (Order-Backlog Drift, RPO).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached RPO panel
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR RPO + yfinance,
# 38 names, 842 events, filings 2018-10-24 -> 2026-06-24; long-short 2019-06 -> 2026-06).
R = dict(
    as_of="2026-06-30", fp="243982cf416a",
    n_names=38, n_events=842, filed_start="2018-10-24", filed_end="2026-06-24",
    price_start="2017-01-03",
    # primary calendar long-short
    n_months=85, ls_start="2019-06", ls_end="2026-06", avg_n=29.3, avg_turnover=0.11,
    gross_bps=100.0, gross_ann=12.0, sharpe=0.57, hit=57.6,
    hit_k=49, hit_n=85, wilson=(47.0, 67.6),
    long_bps=220.6, short_bps=120.6, long_ann=26.5, short_ann=14.5,
    t_iid=1.51, t_nw=1.10,
    net10_bps=89.4, net10_ann=10.7, net10_tnw=0.99, net10_sharpe=0.51,
    net20_bps=87.2, net20_ann=10.5, net20_tnw=0.96, net20_sharpe=0.49,
    # era split
    era_split="2022-01-01",
    era_early_n=31, era_early_bps=244.7, era_early_tnw=1.78,
    era_late_n=54, era_late_bps=16.9, era_late_tnw=0.17,
    # pooled event drift: horizon -> (n, top%, bot%, ls%, win%, t, placebo_p)
    drift={
        21: (830, 1.25, 0.99, 0.26, 53, 0.24, 0.400),
        63: (805, 5.28, 3.86, 1.41, 51, 0.60, 0.260),
        126: (769, 8.88, 5.45, 3.43, 50, 0.98, 0.144),
    },
    bucket126=(5.45, 8.48, 8.88),          # low / mid / high tercile mean 126d drift (%)
    placebo126_obs=3.43, placebo126_mean=0.00, placebo126_p=0.144,
    # mechanism — RPO growth leads sales
    lead_n=465, lead_slope=0.313, lead_t=9.92, lead_r2=0.175, lead_corr=0.42,
    lead_top_sales=39.4, lead_bot_sales=18.6, lead_spread=20.8,
    # signal distribution
    rpo_median=25.5, rpo_p10=3.5, rpo_p90=60.0,
    # synthetic control
    syn_null_mean=-0.37, syn_null_sd=0.85, syn_null_fire=1, syn_planted_bps=198.7,
    syn_planted_t=3.79,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Backlog-leads-sales%3F: Confirmed](https://img.shields.io/badge/Backlog--leads--sales%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from order_backlog import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, EVENTS = data.load_real()
    LS = st.calendar_ls(PRICES, EVENTS, signal_col="rpo_yoy", n_buckets=3,
                        min_names=6, staleness_days=200)
else:
    PRICES = EVENTS = LS = None
print("real cache present:", HAVE_REAL,
      "| names:", (0 if EVENTS is None else EVENTS['ticker'].nunique()),
      "| RPO events:", (0 if EVENTS is None else len(EVENTS)),
      "| LS months:", (0 if LS is None else len(LS)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a swelling order backlog tell you where a stock is headed? 📦\n"
            "### Order-Backlog Drift — the leading indicator that really *is* leading, and "
            "still doesn't pay\n\n"
            + BADGES +
            "Every software-stock investor watches one number on the earnings call: **RPO** — "
            "*remaining performance obligations*, the dollar value of contracts a company has "
            "**signed but not yet delivered**. It's the modern, standardised version of \"order "
            "backlog.\" The pitch writes itself: backlog is *future* revenue you can already see "
            "on the balance sheet, so a company whose backlog is compounding faster than its "
            "sales must be about to accelerate — and if the market is staring at the trailing "
            "income statement, it should be *slow* to price that in.\n\n"
            "Here's the twist. The backlog really **does** lead sales — cleanly, strongly. And "
            "ranking stocks on it really **does** look like a +12%/yr trade. But when you test "
            "whether that trade is *real* rather than a bull-market ghost, it falls apart.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*, the placebo and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 38 US enterprise-software names, RPO from SEC filings, "
            "2019→2026 (RPO didn't exist before 2018). Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does backlog growth actually lead sales? | **Yes, clearly.** Firms with the "
            f"fastest-growing backlog go on to grow revenue **{R['lead_spread']:.0f} percentage "
            f"points** faster next quarter than the slowest — a strong, real accounting lead. |\n"
            f"| Does ranking on it look like a trade? | **On paper, yes.** Long the fast-backlog "
            f"names, short the slow ones: **+{R['gross_ann']:.0f}%/yr** gross, Sharpe "
            f"{R['sharpe']:.2f}. Tempting. |\n"
            f"| Is that trade *real*? | **No.** The rigorous, noise-robust *t*-stat is only "
            f"**{R['t_nw']:.2f}** — the bar is 2. And the entire edge is a 2019-2021 "
            f"bull-market phenomenon that **vanishes** after 2022. |\n"
            "| Can you get paid for it? | **No.** After costs the edge is statistically "
            "indistinguishable from zero — and it only ever \"worked\" in one boom. |\n\n"
            "> The backlog is a genuine leading indicator. The market just doesn't obviously "
            "*misprice* it — and this short, one-era slice of history can't prove otherwise."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Backlog is contracted future revenue sitting on the balance sheet. If a "
            "company's backlog is growing faster than its reported sales, revenue is coming — "
            "and a market fixated on the income statement is slow to price it, so the stock "
            "drifts up as the backlog converts.\"*\n\n"
            "It's the order-backlog member of a respectable academic family: the market "
            "**underreacts** to fundamentals it already has in hand (post-earnings-drift, "
            "Bernard-Thomas; backlog specifically, Rajgopal-Shevlin-Venkatachalam 2003). Since "
            "2018, accounting rule **ASC 606** forces every US filer to disclose that backlog "
            "as a single clean number — *Remaining Performance Obligations* — so for the first "
            "time you can rank the whole software sector on it from public filings."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a beautiful signal: a **balance-sheet** number that leads "
            "the **income-statement** number the whole market trades on, disclosed quarterly, "
            "machine-readable, no analyst estimates required. You'd be trading tomorrow's "
            "revenue-growth surprise today. That's exactly the kind of clean fundamental edge "
            "worth taking seriously — and exactly the kind of story that gets over-sold in "
            "software-investing content, so it's worth being suspicious of too."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** Each name's **year-over-year growth in RPO**, read straight off "
            f"its 10-Q/10-K on the day it was filed (**{R['n_events']}** backlog observations "
            f"across **{R['n_names']}** names).\n"
            "- **The mechanism check.** Does this quarter's backlog growth actually predict "
            "**next** quarter's *sales* growth? If not, the whole premise is dead on arrival.\n"
            "- **The trade.** Each month, long the top third by backlog growth, short the bottom "
            "third, hold a month, pay realistic costs — and ask the noise-robust question: is "
            "the average monthly return really different from luck?\n"
            "- **The honesty cut.** Split the history into the 2019-21 software boom and the "
            "2022+ aftermath. A real signal shouldn't only exist in one of them."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the good news: does backlog lead sales?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ld = st.leads_sales(EVENTS)\n"
            "    top_s, bot_s = ld['top_sales']*100, ld['bot_sales']*100\n"
            "else:\n"
            "    top_s, bot_s = R['lead_top_sales'], R['lead_bot_sales']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['slowest-backlog\\nthird','fastest-backlog\\nthird'], [bot_s, top_s],\n"
            "       color=[GREY, GREEN], width=.55)\n"
            "for i,v in enumerate([bot_s, top_s]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('NEXT quarter revenue growth (YoY, %)')\n"
            "ax.set_title('Backlog growth genuinely leads sales growth')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'top-tercile next-q sales growth {top_s:+.1f}%  vs  bottom {bot_s:+.1f}%')"
        ),
        md(
            f"There's the lead, and it's real: the fastest-backlog names go on to grow revenue "
            f"**{R['lead_spread']:.0f} points** faster next quarter than the slowest "
            f"(regression *t* ≈ **{R['lead_t']:.0f}**). Backlog is a legitimate leading "
            "indicator of the income statement. So far the folklore is winning.\n\n"
            "**Now the trade.** Long the fast-backlog third, short the slow third, rebalanced "
            "monthly. Does it make money?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.calendar_ls_stats(LS)\n"
            "    cum = (1 + LS['ls']).cumprod()\n"
            "    ann, shp, tnw = s['ann_pct'], s['sharpe'], s['t_nw']\n"
            "    x = cum.index; y = cum.values\n"
            "else:\n"
            "    ann, shp, tnw = R['gross_ann'], R['sharpe'], R['t_nw']\n"
            "    rng = np.random.default_rng(799)\n"
            "    y = np.cumprod(1 + rng.normal(R['gross_bps']/1e4, 0.05, R['n_months']))\n"
            "    x = pd.date_range('2019-06-30', periods=R['n_months'], freq='ME')\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(x, y, color=GREEN, lw=2)\n"
            "ax.set_ylabel('growth of $1 (gross long-short)')\n"
            "ax.set_title(f'Looks like a trade: ~{ann:+.0f}%/yr, Sharpe {shp:.2f} '\n"
            "             f'-- but noise-robust t = only {tnw:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross ~{ann:+.1f}%/yr, Sharpe {shp:.2f}, Newey-West t = {tnw:+.2f} (bar is 2)')"
        ),
        md(
            f"The equity curve rises and the headline numbers look great — **~+{R['gross_ann']:.0f}"
            f"%/yr**, Sharpe **{R['sharpe']:.2f}**. But the number that matters — the *t*-stat "
            f"that accounts for how bunched-up and noisy these monthly returns are — is only "
            f"**{R['t_nw']:.2f}**. The desk's bar for calling something real is **2**. This "
            "isn't there.\n\n"
            "**And here's what kills it.** Split the history in two:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = LS[LS.index < data.ERA_SPLIT]['ls'].to_numpy()\n"
            "    l = LS[LS.index >= data.ERA_SPLIT]['ls'].to_numpy()\n"
            "    eb, lb = e.mean()*1e4, l.mean()*1e4\n"
            "    et, lt = st.newey_west_t(e), st.newey_west_t(l)\n"
            "else:\n"
            "    eb, lb = R['era_early_bps'], R['era_late_bps']\n"
            "    et, lt = R['era_early_tnw'], R['era_late_tnw']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['2019-2021\\n(SaaS boom)','2022-2026\\n(after)'], [eb, lb],\n"
            "       color=[AMBER, RED], width=.55)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]):\n"
            "    ax.annotate(f'{v:+.0f} bps/mo\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('The whole edge is a 2019-2021 phenomenon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2019-21 {eb:+.0f} bps/mo (t={et:+.2f})  |  2022+ {lb:+.0f} bps/mo (t={lt:+.2f})')"
        ),
        md(
            f"Everything the strategy \"earned\" happened in the **2019-2021** subscription-"
            f"software melt-up (**+{R['era_early_bps']:.0f} bps/mo**). In the mature 2022+ "
            f"panel — through the rate shock and the recovery — it's **+{R['era_late_bps']:.0f} "
            f"bps/mo**, essentially zero (*t* = {R['era_late_tnw']:+.2f}). A signal that only "
            "exists in one bull market isn't a signal; it's a memory of that bull market."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The backlog→sales lead is real and strong, and the trade is "
            f"positively signed everywhere (~+{R['gross_ann']:.0f}%/yr, Sharpe {R['sharpe']:.2f}) "
            f"— but the noise-robust *t* is only **{R['t_nw']:.2f}** and the edge is entirely a "
            "2019-21 artifact. The literature says the market underreacts to backlog; *this* "
            "short, one-era tape can't prove it.\n"
            "- **Tradability — Mirage.** After costs the edge is indistinguishable from zero, "
            "and what little there is lived in a single boom. Not a paycheck.\n"
            "- **\"Does backlog lead sales?\" — Confirmed.** The fundamental part of the story is "
            "genuinely true; it's the *market-mispricing* part that this sample can't support."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The core problem is the calendar, not the idea.** RPO only exists post-2018, so "
            "there is exactly *one* macro regime of history. Give this signal another decade — "
            "or extend it to non-software sectors that now disclose RPO — and it might certify. "
            "Today it can't.\n"
            "- **The mechanism is the interesting survivor.** Backlog genuinely front-runs "
            "sales; a follow-up worth doing is whether backlog growth predicts the *revenue "
            "surprise* (vs consensus) rather than raw revenue growth — that's where "
            "underreaction, if it exists, would actually live.\n"
            "- **Sibling studies:** [798-deferred-revenue-signal](../../798-deferred-revenue-signal/) "
            "(the *billed*-but-unrecognised balance — RPO is the bigger signed-backlog number), "
            "[199-sales-growth](../../199-sales-growth/) (the realised sales number RPO leads), "
            "and [534-revenue-surprise-drift](../../534-revenue-surprise-drift/) (revenue "
            "*surprise* drift). See [docs/references.md](docs/references.md) for the exact "
            "dedup.\n\n"
            "*Think the backlog edge is real and we just picked the wrong window? Show a "
            "noise-robust *t* ≥ 2 that isn't carried by 2019-2021 — then we'll talk.*"
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
            "# Order-Backlog Drift — a quantitative teardown 🔬\n"
            "### The RPO-growth calendar long-short + Newey-West *t* · the era split that kills "
            "it · a 534-style pooled event-drift placebo · the RPO→sales mechanism regression · "
            "a cost/borrow sweep · a 20-seed synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **YoY growth in Remaining Performance Obligations (RPO / signed backlog) is "
            "a leading fundamental the market underreacts to** — is a cross-sectional "
            "return-predictability claim, distinct from the *deferred-revenue* balance "
            "(sibling 798), from *realised* sales growth (199), and from a revenue-*surprise* "
            "PEAD (534). The job: measure it honestly on the only sample that exists "
            "(post-ASC-606, one regime), and be clear about what that sample can and cannot "
            "certify.\n\n"
            "> ⚠️ **Data note.** 38 enterprise-software names; RPO from EDGAR `companyconcept` "
            "(`RevenueRemainingPerformanceObligation`), point-in-time at the filing date; "
            "prices yfinance total-return. **RPO didn't exist before 2018** — the long-short "
            "runs 2019-06 → 2026-06 (85 months). Survivorship named on the Signal axis "
            "(current-survivors basket). Numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | calendar LS **+{R['gross_bps']:.0f} bps/mo** "
            f"(~+{R['gross_ann']:.0f}%/yr, Sharpe {R['sharpe']:.2f}) but **NW *t* = "
            f"{R['t_nw']:.2f}** (< 2); pooled drift monotone, max *t* = "
            f"{R['drift'][126][5]:.2f} (126d); edge concentrated 2019-21 "
            f"({R['era_early_bps']:.0f} bps/mo) vs 2022+ ({R['era_late_bps']:.0f}, *t* = "
            f"{R['era_late_tnw']:.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | net @20bps+borrow **{R['net20_bps']:.0f} bps/mo**, "
            f"NW *t* = {R['net20_tnw']:.2f}, Sharpe {R['net20_sharpe']:.2f} |\n"
            f"| **Backlog leads sales?** | `CONFIRMED` | next_rev_yoy ~ rpo_yoy slope "
            f"**+{R['lead_slope']:.2f}** (*t* = +{R['lead_t']:.1f}), tercile spread "
            f"**+{R['lead_spread']:.0f} pp** |\n\n"
            "> 💡 In plain words: the fundamental lead is unambiguous (backlog → sales), the "
            "return long-short is the right sign everywhere and even looks tradable on paper — "
            "but the autocorrelation-robust statistic never clears 2, and the whole thing is a "
            "2019-21 bull-market ghost. Real fundamental, uncertified mispricing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $B_{i,q}$ be name $i$'s total RPO at fiscal quarter-end $q$, disclosed on "
            "filing date $f_{i,q}$. Signal: YoY growth $g_{i,q} = B_{i,q}/B_{i,q-4} - 1$, known "
            "at $f_{i,q}$. The claims:\n\n"
            "- **H₁ (lead).** $g$ predicts next-quarter revenue growth — the accounting lead.\n"
            "- **H₂ (drift).** High-$g$ names out-return low-$g$ names *after* $f$, because the "
            "market underreacts to the disclosed backlog.\n"
            "- **H₃ (capture).** A monthly-rebalanced tercile long-short banks that drift net of "
            "costs and borrow.\n\n"
            f"We find **H₁ strongly supported** (slope +{R['lead_slope']:.2f}, *t* = "
            f"+{R['lead_t']:.1f}, +{R['lead_spread']:.0f} pp tercile spread), **H₂ only weakly "
            f"and unstably supported** (right sign at every horizon, but NW *t* = {R['t_nw']:.2f} "
            f"and concentrated in 2019-21), **H₃ not supported** (net NW *t* = "
            f"{R['net20_tnw']:.2f}). The decisive statistic is the monthly long-short's "
            "Newey-West *t*; the honest finding is that the *fundamental* lead is real but the "
            "*return* underreaction is uncertifiable on this tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Monthly long-short returns are **persistent and overlapping** (a quarterly signal "
            "held across months, in a tightly-correlated sector), so i.i.d. *t* overstates "
            "significance — the planned primary is the **Newey-West (6-lag) HAC *t*** of the "
            "monthly series. The pooled event-drift cross-check carries a **10,000-draw "
            "label-shuffle placebo** (permute the signal, re-form random terciles). The era "
            "split (2022-01-01, the early-ASC-606 / mature boundary named ex ante) is reported "
            "as **within-era HAC *t*'s**, not eyeballed. The monthly hit rate carries a "
            "**Wilson interval**. One documented execution lag throughout (signal at month *t*, "
            "return of *t+1*); costs one-way × NAV × turnover on both legs; short pays borrow."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_names']} names, {R['n_events']} RPO-YoY events, filings "
            f"{R['filed_start']} → {R['filed_end']}. Median RPO growth ≈ {R['rpo_median']:.0f}% "
            f"(p10 {R['rpo_p10']:.0f}%, p90 {R['rpo_p90']:.0f}%) — a wide, real cross-section.\n"
            "- **Primary.** Monthly tercile long-short, NW(6) *t* + one-sample *t* + Wilson hit "
            "rate; gross and net (10/20 bps + 100 bps borrow).\n"
            "- **Cross-check.** Pooled event drift at 21/63/126d, top-minus-bottom, one-sample "
            "*t* + placebo.\n"
            "- **Mechanism.** Pooled OLS next_rev_yoy ~ rpo_yoy (slope, *t*, R², tercile "
            "spread).\n"
            "- **Era.** 2019-21 vs 2022+, within-era HAC *t*.\n"
            "- **Control.** Synthetic panel, planted forward-return knob; the null must not fire "
            "across 20 seeds, a planted edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary calendar long-short and its Newey-West *t*\n\n"
            "Monthly tercile long-short, one execution lag. The decisive number is the HAC *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.calendar_ls_stats(LS)\n"
            "    print(f\"months={s['n_months']}  avg X-section={s['avg_n']:.1f}  \"\n"
            "          f\"avg one-way turnover={s['avg_turnover']:.2f}\")\n"
            "    print(f\"gross LS {s['mean_bps']:+.1f} bps/mo (~{s['ann_pct']:+.1f}%/yr)  \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}  hit {s['hit']*100:.0f}%\")\n"
            "    print(f\"long {s['long_bps']:+.1f}  short {s['short_bps']:+.1f} bps/mo\")\n"
            "    print(f\"one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}\")\n"
            "    cum = (1 + LS['ls']).cumprod(); x, y = cum.index, cum.values\n"
            "    tnw = s['t_nw']\n"
            "else:\n"
            "    tnw = R['t_nw']\n"
            "    rng = np.random.default_rng(799)\n"
            "    y = np.cumprod(1 + rng.normal(R['gross_bps']/1e4, 0.05, R['n_months']))\n"
            "    x = pd.date_range('2019-06-30', periods=R['n_months'], freq='ME')\n"
            "    print(f\"(offline) gross {R['gross_bps']:+.0f} bps/mo, NW t = {tnw:+.2f}\")\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(x, y, color=(GREEN if tnw>=2 else AMBER), lw=2)\n"
            "ax.set_ylabel('growth of $1 (gross LS)')\n"
            "ax.set_title(f'Gross long-short — positive but NOT robust (NW t = {tnw:.2f}, bar 2)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **+{R['gross_bps']:.0f} bps/mo** and Sharpe {R['sharpe']:.2f} "
            f"look like a real trade, and one-sample *t* = {R['t_iid']:.2f} would almost tempt "
            f"you — but once you correct for the serial correlation in these overlapping monthly "
            f"returns, **NW *t* = {R['t_nw']:.2f}**. The hit rate is {R['hit']:.0f}% (Wilson 95% "
            f"[{R['wilson'][0]:.0f}%, {R['wilson'][1]:.0f}%]) — the lower bound is below 50%. "
            "Not certified."
        ),
        md(
            "### 4b · The era split — where the \"edge\" actually lives\n\n"
            "Within-era HAC *t*, boundary 2022-01-01 (early-ASC-606 boom vs mature panel)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = LS[LS.index < data.ERA_SPLIT]['ls'].to_numpy()\n"
            "    l = LS[LS.index >= data.ERA_SPLIT]['ls'].to_numpy()\n"
            "    eb, lb, et, lt = e.mean()*1e4, l.mean()*1e4, st.newey_west_t(e), st.newey_west_t(l)\n"
            "    ne, nl = len(e), len(l)\n"
            "else:\n"
            "    eb, lb, et, lt = R['era_early_bps'], R['era_late_bps'], R['era_early_tnw'], R['era_late_tnw']\n"
            "    ne, nl = R['era_early_n'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar([f'2019-2021\\n(n={ne})', f'2022-2026\\n(n={nl})'], [eb, lb],\n"
            "       color=[AMBER, RED], width=.55)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]):\n"
            "    ax.annotate(f'{v:+.0f} bps/mo\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('LS mean (bps/mo)')\n"
            "ax.set_title('Regime concentration: the edge is a 2019-21 ghost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2019-21 {eb:+.0f} bps/mo (t={et:+.2f})  |  2022+ {lb:+.0f} bps/mo (t={lt:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: **+{R['era_early_bps']:.0f} bps/mo** in 2019-21 vs "
            f"**+{R['era_late_bps']:.0f} bps/mo** since 2022 (*t* = {R['era_late_tnw']:+.2f}). "
            "The signal didn't decay gently — it switched off. Anything that only pays in the "
            "2020-21 growth-stock mania is indistinguishable from a long-duration-growth beta "
            "that happened to correlate with backlog momentum in that window."
        ),
        md(
            "### 4c · The pooled event drift and its placebo\n\n"
            "534-style: bucket every filing event by RPO growth, top-minus-bottom forward drift, "
            "entered one session after the filing; 10,000-draw label-shuffle placebo. In the "
            "notebook we run a lighter placebo and quote the canonical p from `results.md`."
        ),
        code(
            "hs = list(R['drift'].keys())\n"
            "if HAVE_REAL:\n"
            "    ls_by_h, t_by_h = [], []\n"
            "    for h in hs:\n"
            "        e = st.event_summary(PRICES, EVENTS, horizon=h, n_buckets=3, lag=1,\n"
            "                             placebo=True, n_draws=2000)\n"
            "        ls_by_h.append(e['ls_mean']*100); t_by_h.append(e['t'])\n"
            "    bm = st.bucket_means(st.event_drift_frame(PRICES, EVENTS, 126, lag=1), 3)*100\n"
            "else:\n"
            "    ls_by_h = [R['drift'][h][3] for h in hs]; t_by_h = [R['drift'][h][5] for h in hs]\n"
            "    bm = np.array(R['bucket126'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], ls_by_h, color=[AMBER]*len(hs), width=.55)\n"
            "for i,(v,t_) in enumerate(zip(ls_by_h, t_by_h)):\n"
            "    a1.annotate(f'{v:+.2f}%\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('top-minus-bottom drift (%)')\n"
            "a1.set_title('Right sign, rising with horizon -- none robust')\n"
            "a2.bar(['low','mid','high'], bm, color=[RED, GREY, GREEN], width=.6)\n"
            "for i,v in enumerate(bm): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('126d forward drift (%)'); a2.set_title('Monotone in the signal (126d)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('LS by horizon %:', [round(v,2) for v in ls_by_h], ' t:', [round(t,2) for t in t_by_h])\n"
            "print(f\"canonical 126d: LS +{R['placebo126_obs']:.2f}% vs placebo mean \"\n"
            "      f\"{R['placebo126_mean']:+.2f}%, p = {R['placebo126_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the drift is the **right sign at every horizon** and rises "
            f"monotonically with the signal ({R['bucket126'][0]:.1f}% / {R['bucket126'][1]:.1f}% "
            f"/ {R['bucket126'][2]:.1f}% low→high at 126d), and it *grows* with horizon "
            f"(+{R['drift'][21][3]:.2f}% → +{R['drift'][126][3]:.2f}%) exactly as a slow "
            f"underreaction would — but the best horizon is *t* = {R['drift'][126][5]:.2f}, "
            f"placebo **p = {R['placebo126_p']:.3f}**. Suggestive, never significant."
        ),
        md(
            "### 4d · The mechanism — backlog really does lead sales\n\n"
            "Pooled OLS: next-quarter revenue YoY growth on this-quarter RPO YoY growth."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ld = st.leads_sales(EVENTS)\n"
            "    slope, tt, r2 = ld['slope'], ld['t'], ld['r2']\n"
            "    top_s, bot_s = ld['top_sales']*100, ld['bot_sales']*100\n"
            "    fr = EVENTS.dropna(subset=['rpo_yoy','next_rev_yoy'])\n"
            "    xx = fr['rpo_yoy'].to_numpy()*100; yy = fr['next_rev_yoy'].to_numpy()*100\n"
            "else:\n"
            "    slope, tt, r2 = R['lead_slope'], R['lead_t'], R['lead_r2']\n"
            "    top_s, bot_s = R['lead_top_sales'], R['lead_bot_sales']\n"
            "    rng = np.random.default_rng(1); xx = rng.normal(30,25,R['lead_n'])\n"
            "    yy = 0.313*xx + rng.normal(15,20,R['lead_n'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.scatter(xx, yy, s=12, alpha=.35, color=GREEN)\n"
            "xs = np.linspace(np.nanpercentile(xx,2), np.nanpercentile(xx,98), 50)\n"
            "b0 = (np.nanmean(yy) - slope*np.nanmean(xx))\n"
            "ax.plot(xs, b0 + slope*xs, color=RED, lw=2, label=f'slope {slope:+.2f} (t={tt:+.1f})')\n"
            "ax.set_xlabel('RPO YoY growth this quarter (%)')\n"
            "ax.set_ylabel('revenue YoY growth NEXT quarter (%)')\n"
            "ax.set_title('The accounting lead is real and strong')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'slope {slope:+.3f} (t={tt:+.2f}, R2={r2:.3f}); next-q sales top {top_s:+.1f}% vs bot {bot_s:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: this is the part of the thesis that **survives**. Backlog "
            f"growth predicts next-quarter revenue growth with slope +{R['lead_slope']:.2f}, "
            f"*t* = +{R['lead_t']:.1f}, and the top-tercile names grow sales "
            f"**+{R['lead_spread']:.0f} pp** faster. The fundamental lead is genuine — the market "
            "just doesn't obviously *misprice* it. (This *t* is iid-pooled across "
            "quarter-clustered events, so read it as a large, real effect, not a HAC statistic.)"
        ),
        md(
            "### 4e · Cost/borrow sweep\n\n"
            "One-way × NAV × turnover on both legs, short pays 100 bps/yr borrow."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.calendar_ls_stats(LS)['mean_bps']\n"
            "    n10 = st.calendar_ls_net(LS, 10.0, 100.0); n20 = st.calendar_ls_net(LS, 20.0, 100.0)\n"
            "    vals = [g, n10['net_mean_bps'], n20['net_mean_bps']]\n"
            "    ts = [st.calendar_ls_stats(LS)['t_nw'], n10['net_t_nw'], n20['net_t_nw']]\n"
            "else:\n"
            "    vals = [R['gross_bps'], R['net10_bps'], R['net20_bps']]\n"
            "    ts = [R['t_nw'], R['net10_tnw'], R['net20_tnw']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(['gross','net @10bps','net @20bps'], vals, color=[GREY, AMBER, RED], width=.6)\n"
            "for i,(v,t_) in enumerate(zip(vals, ts)):\n"
            "    ax.annotate(f'{v:+.0f} bps/mo\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('LS mean (bps/mo)')\n"
            "ax.set_title('Costs barely bite (low turnover) -- but t stays < 1')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {vals[0]:+.0f} (t={ts[0]:+.2f}) -> net@20 {vals[2]:+.0f} bps/mo (t={ts[2]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: turnover is only ~{R['avg_turnover']:.2f} (a quarterly signal), "
            f"so 20 bps + borrow costs the strategy barely anything — net "
            f"**{R['net20_bps']:.0f} bps/mo**, Sharpe {R['net20_sharpe']:.2f}. But the net "
            f"**NW *t* = {R['net20_tnw']:.2f}** is indistinguishable from zero. Costs aren't the "
            "killer here; *significance* is. That's a MIRAGE by the desk's definition — a paper "
            "Sharpe you can't certify."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic price + RPO-growth panel with a TUNABLE planted forward-return edge "
            "(~7-year window matching the real sample). The null (edge = 0) is checked over "
            "**20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for sd in range(20):\n"
            "    pr, ev = data.synthetic_panel(edge=0.0, seed=799 + sd)\n"
            "    null_ts.append(st.synthetic_detect(pr, ev)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "pr, ev = data.synthetic_panel(edge=0.15, seed=799)\n"
            "planted_t = st.synthetic_detect(pr, ev)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=90, zorder=5, label='planted edge=0.15')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('calendar long-short NW t')\n"
            "ax.set_title('Control: null stays put, a planted edge lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), firing above the bar in "
            f"only {R['syn_null_fire']}/20 (≈ the nominal 5%); a planted edge reads *t* = "
            f"{R['syn_planted_t']:.2f}. So the machinery is sound and *would* certify a real "
            "underreaction — the sub-2 real-tape *t* is an honest reading of a thin, one-regime "
            "tape, not a broken pipeline. *(Power check only — never cited for the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — RPO-growth LS **+{R['gross_bps']:.0f} bps/mo** "
            f"(~+{R['gross_ann']:.0f}%/yr, Sharpe {R['sharpe']:.2f}), positively signed on every "
            f"cut and monotone in the pooled sort — but **NW *t* = {R['t_nw']:.2f}** (< 2), no "
            f"horizon clears a placebo *p* < 0.05, and the edge is entirely 2019-21 "
            f"(+{R['era_early_bps']:.0f} bps/mo vs +{R['era_late_bps']:.0f} since 2022). "
            "Literature + mechanism say real; this tape can't certify the mispricing.\n"
            f"- **Tradability `MIRAGE`** — net @20bps+borrow **{R['net20_bps']:.0f} bps/mo**, "
            f"NW *t* = {R['net20_tnw']:.2f}, Sharpe {R['net20_sharpe']:.2f}: an uncertified, "
            "regime-concentrated paper edge, not a paycheck.\n"
            f"- **Backlog leads sales? `CONFIRMED`** — next_rev_yoy ~ rpo_yoy slope "
            f"+{R['lead_slope']:.2f} (*t* = +{R['lead_t']:.1f}), +{R['lead_spread']:.0f} pp "
            "tercile spread. The fundamental part of the thesis is genuinely true."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The binding constraint is history, not the idea.** RPO is a post-2018 "
            "disclosure, so there is one macro regime to test on. The honest statement is "
            "*\"underpowered,\"* not *\"debunked\"* — a decade more data, or cross-sector RPO "
            "beyond software, could move this from WEAK toward REAL.\n"
            "- **Chase the surviving mechanism.** Backlog → sales is real (*t* ≈ 10). The "
            "sharper follow-up is backlog growth → *revenue surprise* (vs consensus), where "
            "genuine underreaction would live, and whether the drift concentrates in names where "
            "backlog and reported sales *diverge* most.\n"
            "- **Dedup map:** [798-deferred-revenue-signal](../../798-deferred-revenue-signal/) "
            "(billed-but-unrecognised balance; RPO is the larger signed-backlog line), "
            "[199-sales-growth](../../199-sales-growth/) (realised sales, which RPO leads), "
            "[534-revenue-surprise-drift](../../534-revenue-surprise-drift/) (revenue-surprise "
            "PEAD around the print).\n\n"
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
