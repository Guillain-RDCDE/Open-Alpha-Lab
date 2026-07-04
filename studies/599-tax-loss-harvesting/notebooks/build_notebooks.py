"""Generate the two narrative notebooks for Study 599 (Tax-Loss Harvesting).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/IVV closes
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic control runs anywhere with no network. Heavy sweeps (rate-grid
cohorts, decay profile, thresholds) are quoted from ``R`` — ``examples/verify.py`` recomputes
them all.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (SPY total-return 1993-01-29 ->
# 2026-06-30, lot-level TLH engine, 2 bps one-way costs, one-day execution lag).
R = dict(
    start="1993-01-29", end="2026-06-30", n_days=8411, years=33.4,
    fingerprint="cb2440086a51",
    twin_corr=0.9932, twin_gap_bps=0.01, twin_te_bps=223, twin_days=6565,
    # full-period lump-sum at 35/15 — the lock-up punchline
    lump_harvests=0, lump_alpha=0.0,
    # contributor grid ($10k + $1k/mo): label -> (alpha bps/yr, terminal uplift %, harvests)
    grid=[("35/15 (top bracket, ST offsets)", 13.1, 4.49, 667),
          ("24/15 (middle bracket)", 8.8, 3.00, 667),
          ("15/15 (deferral only)", 5.4, 1.82, 667),
          ("35/15 + step-up at death", 13.9, 4.75, 667),
          ("0/0 (0% bracket — myth check)", -0.4, -0.13, 667),
          ("0/0 + step-up (low-bracket donor)", -0.4, -0.13, 667)],
    harvested_total=55216, st_share=88.5, tax_saved=18057,
    term_tax_tlh=544694, term_tax_bh=511804,
    # 10-year lump-sum quarterly cohorts at 35/15 (the Signal axis)
    n_cohorts=94, co_mean=46.0, co_median=23.5, co_sd=53.9, co_min=-0.4, co_max=189.0,
    hac_t=2.88, hac_lag=40, share_pos=73.4, zero_harvest=24,
    best_start="2007-10-01", worst_start="2010-10-01",
    decade_means="1990s +13.2, 2000s +90.8, 2010s +12.5 bps/yr",
    # cohort rate grid: (rates, mean, median, HAC t)
    co_grid=[("35/15", 46.0, 23.5, 2.88), ("24/15", 27.5, 14.7, 2.78),
             ("15/15", 12.3, 6.2, 2.42)],
    # path dependence (contributor run): bear vs calm harvest yield, % of NAV per year
    bear_mean=2.23, calm_mean=0.16, welch_t=2.72, welch_dof=6.1,
    bear_years=(2000, 2001, 2002, 2008, 2009, 2020, 2022),
    rich_years="2008: 4.80%, 2002: 4.29%, 2009: 2.95%, 2001: 2.93%, 1994: 1.33%, 2003: 0.90%",
    # decay: harvested loss as % of initial stake by year since inception (10y cohorts)
    decay=[6.57, 3.19, 2.58, 1.81, 1.60, 0.46, 0.19, 0.15, 0.15, 0.17],
    decay_tail=0.17,
    # thresholds: (harvest threshold %, cohort mean, HAC t, harvests/cohort)
    thresholds=[(0.5, 45.7, 2.94, 4.3), (1.0, 46.0, 2.88, 4.3), (5.0, 48.8, 2.93, 2.5)],
    # synthetic control (20 seeds/cell): (sigma, alpha 35/15, sd, alpha rates 0/0)
    syn=[(0.02, 0.09, 0.4, -0.02), (0.18, 47.00, 47.7, -2.67),
         (0.35, 128.45, 104.5, -4.33)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Who_gets_zero%3F: Confirmed](https://img.shields.io/badge/Who_gets_zero%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab)
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from tax_loss_harvesting import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY, IVV = data.load_real()
else:
    SPY = IVV = None
print("real SPY cache present:", HAVE_REAL,
      "| days:", (0 if SPY is None else len(SPY)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


def _grid_md_table():
    rows = "\n".join(f"| {lab} | **{a:+.1f}** | {u:+.2f}% |" for lab, a, u, _ in R["grid"])
    return ("| rate pair (ST/LT) | after-tax alpha (bps/yr) | terminal uplift |\n"
            "|---|--:|--:|\n" + rows)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Tax-loss harvesting — the \"free money\" your robo-advisor promises 🧾\n"
            "### Real arithmetic, not market magic — and for some people, exactly zero\n\n"
            + BADGES +
            "Every robo-advisor pitch says the same thing: *\"our software watches your account "
            "every day, sells anything that's down, instantly buys a **twin fund** so you stay "
            "invested, and the harvested loss cuts your tax bill — an extra **0.3–1% a year**, "
            "for free.\"*\n\n"
            "It sounds like a perpetual-motion machine. It isn't — but it isn't a scam either. "
            "It's **tax arithmetic**: you stay 100% invested the whole time (the twin fund moves "
            "with the market just like the original), so the market can't be where the money comes "
            "from. The money comes from **your own tax rates** — and that has three consequences "
            "this notebook walks through: the gain is real, it's **front-loaded**, and if your tax "
            "rate is zero, your share is *exactly nothing minus costs*.\n\n"
            "> 📓 **Plain-language layer.** Want the lot-level engine, the HAC *t*-stats and the "
            "rate grid? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Scope note.** We test the claim on a single broad-index fund (SPY, with IVV as "
            "the twin) — the plain-vanilla robo setup. Stock-level \"direct indexing\" harvests "
            "more (see the references). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md). *Not tax advice.*"
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does harvesting add after-tax return? | **Yes — really.** The mean 10-year cohort "
            f"banks **+{R['co_mean']:.0f} bps/yr** at top-bracket rates (statistically solid — the "
            f"quants notebook has HAC *t* = +{R['hac_t']:.2f}). |\n"
            f"| Is it the advertised 0.3–1%/yr? | **Only at the top.** The *median* cohort gets "
            f"**+{R['co_median']:.0f} bps/yr**, and a 33-year steady saver nets **+13 bps/yr**. |\n"
            f"| Where does the money come from? | **Your own tax rates** — deduct losses at ~35%, "
            f"repay gains later at ~15%, plus deferral. Set both rates to zero and the \"alpha\" is "
            f"**−0.4 bps/yr** (pure trading costs). |\n"
            f"| Does it last? | **No — it fades.** Year-1 harvest ≈ **6.6%** of your stake; by years "
            f"7–10 it's ≈ **{R['decay_tail']:.2f}%/yr**. Old money stops harvesting. |\n"
            f"| Who gets zero? | The **0%-bracket** retiree or donor — including with step-up at "
            f"death. A quarter of 10-year cohorts never harvest *at all*. |"
        ),

        md(
            "## What a \"harvest\" actually is\n\n"
            "Say you bought an S&P 500 fund at \\$100 and it drops to \\$90. You now own the same "
            "market exposure either way — but if you **sell**, the \\$10 loss becomes a *realised* "
            "loss the tax code lets you deduct against gains elsewhere. So the machine sells, "
            "immediately buys a **twin fund** (a different issuer's S&P 500 ETF — SPY↔IVV — so the "
            "*wash-sale rule* doesn't void the loss), and you keep riding the market as if nothing "
            "happened. At ~35% short-term rates, that \\$10 loss puts **\\$3.50** back in your "
            "pocket today.\n\n"
            "The catch nobody advertises: your new lot's cost basis is now **\\$90**, so when you "
            "finally sell, your taxable gain is \\$10 **bigger**. You didn't erase tax — you "
            "**moved it in time** (deferral) and **changed its rate** (deduct at 35%, repay at "
            "15%). That spread — not the market — is the entire product.\n\n"
            "Below: the tape we run it on — every day from 1993 to 2026, a lot-level simulator "
            "checks a steady saver's account ($10k start + $1k/month) and harvests any lot down "
            "more than 1%."
        ),
        code(
            "# the harvest calendar — when does the machine actually find losses?\n"
            "if HAVE_REAL:\n"
            "    r = st.after_tax_alpha(SPY, initial=10_000, monthly_contrib=1_000,\n"
            "                           st_rate=0.35, lt_rate=0.15)\n"
            "    hy = st.harvest_yield_by_year(r['h'])\n"
            "    fig, ax = plt.subplots()\n"
            "    colors = [RED if y in R['bear_years'] else GREY for y in hy.index]\n"
            "    ax.bar(hy.index, hy.values, color=colors)\n"
            "    ax.set_title('Harvested losses per calendar year (% of account value) — '\n"
            "                 'red = bear-market years')\n"
            "    ax.set_ylabel('% of avg NAV harvested')\n"
            "    print(f\"after-tax alpha {r['alpha_bps']:+.1f} bps/yr over {r['years']:.1f} years \"\n"
            "          f\"| {r['h']['n_harvests']} harvests | tax savings reinvested \"\n"
            "          f\"${r['h']['tax_saved']:,.0f}\")\n"
            "else:\n"
            "    print('cache missing — quoting docs/results.md: alpha +13.1 bps/yr, 667 harvests')\n"
        ),
        md(
            "**Read the chart:** the machine feasts in **2000–02, 2008–09** (and nibbles in 1994, "
            "2020, 2022) and starves in every calm bull year — 2013, 2017, 2021, 2024 harvest "
            "essentially **nothing**. Tax-loss harvesting is a **bear-market crop**: the quants "
            "notebook shows bear years yield **~14×** the calm years "
            f"({R['bear_mean']:.1f}% vs {R['calm_mean']:.2f}% of NAV per year).\n\n"
            "> 🔬 **For the quants.** The engine is lot-level: every contribution, harvest "
            "repurchase and reinvested tax saving opens its own lot with its own basis and clock; "
            "losses are classified short/long-term; execution is at the **next** close (one lag); "
            "2 bps one-way costs per leg; both books pay per-lot capital-gains tax at the end."
        ),

        md(
            "## Where the money comes from — run the same tape at different tax rates\n\n"
            "Same market, same 667 harvests, same everything — only the investor's tax bracket "
            "changes:\n\n" + _grid_md_table() + "\n\n"
            "The market path is identical in every row. The **entire** payoff is the rate pair: a "
            "top-bracket investor with short-term gains to offset keeps ~13 bps/yr; equal rates "
            "(pure deferral) keep ~5; a **0% bracket keeps less than nothing** (the trading costs). "
            "This is the cleanest possible proof that TLH is *arithmetic, not alpha*."
        ),
        code(
            "labels = [g[0] for g in R['grid']]\n"
            "vals = [g[1] for g in R['grid']]\n"
            "cols = [GREEN if v > 10 else (AMBER if v > 0 else RED) for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.barh(range(len(vals))[::-1], vals, color=cols)\n"
            "ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labels, fontsize=9)\n"
            "ax.axvline(0, color='k', lw=.8)\n"
            "ax.set_xlabel('after-tax alpha, bps/yr (33-year contributor, same market path)')\n"
            "ax.set_title('The \"alpha\" is your tax bracket — zero bracket, zero alpha')\n"
            "plt.tight_layout()\n"
        ),

        md(
            "## The machine goes quiet — basis lock-up\n\n"
            "A loss needs a price **below what you paid**. After a few years of normal market "
            "growth, everything you own sits far above its purchase price, and even a nasty "
            "crash (2020: −34% in a month!) barely dents lots bought years earlier. Fresh cash "
            "is the only thing that keeps the machine fed — which is why the robo pitch quietly "
            "assumes you keep contributing forever. A **1993 lump sum never harvests again** "
            "after its first weeks: zero harvests in 33 years."
        ),
        code(
            "yrs = np.arange(1, 11)\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(yrs, R['decay'], color=[GREEN]+[AMBER]*4+[RED]*5)\n"
            "ax.set_xticks(yrs)\n"
            "ax.set_xlabel('year since you invested (average across 94 ten-year cohorts)')\n"
            "ax.set_ylabel('losses harvested, % of initial stake')\n"
            "ax.set_title('The harvest fades as your basis locks up: '\n"
            "             f\"{R['decay'][0]:.1f}% in year 1 → {R['decay_tail']:.2f}%/yr by years 7-10\")\n"
        ),

        md(
            "## The verdict\n\n"
            f"**Real** — the after-tax gain is genuine arithmetic and shows up on 33 years of real "
            f"tape: **+{R['co_mean']:.0f} bps/yr** on the mean 10-year cohort at top-bracket rates, "
            f"statistically solid. **Fragile** — it's front-loaded, bear-market-fed (24 of "
            f"{R['n_cohorts']} cohorts get *nothing*), needs short-term gains elsewhere to offset, "
            f"and the median cohort's **+{R['co_median']:.0f} bps** is *less than the 25 bps fee* "
            "many robos charge to do it for you. **Who gets zero? — confirmed:** at a 0% tax rate "
            "the machine produces exactly **−0.4 bps/yr** (its own trading costs), step-up or not.\n\n"
            "**Take-away:** if you're in a high bracket, harvest — especially in the first years "
            "and in crashes, and ideally yourself, for free. If you're in a low bracket, donating, "
            "or holding until step-up: the robo is harvesting **its fee**, not your alpha.\n\n"
            "*Full stats, engine spec and robustness: "
            "[02_for_the_quants.ipynb](02_for_the_quants.ipynb) · numbers: "
            "[docs/results.md](../docs/results.md) · sources: "
            "[docs/references.md](../docs/references.md). Not investment or tax advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    return nb


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Tax-Loss Harvesting — lot-level engine, cohort inference, rate grid 🧾\n\n"
            + BADGES +
            "**Claim under test:** systematically harvesting losses into a twin fund adds "
            "**0.3–1%/yr** of after-tax alpha (Wealthfront/Betterment; Chaudhuri-Burnham-Lo 2020 "
            "find ~1.08%/yr at *stock level*).\n\n"
            "**Engine spec** (single broad-index implementation, the plain robo setup):\n"
            "* SPY daily **total-return** closes 1993-01-29 → 2026-06-30 (as-of; June 2026 last "
            "complete month). Twin equivalence justified against IVV.\n"
            "* **Lot-level accounting** — every purchase (lump sum, monthly contribution, harvest "
            "repurchase, reinvested tax saving) opens a lot with its own basis + clock.\n"
            "* Daily check: lot below basis by > **1%** and held ≥ **31 calendar days** "
            "(wash-sale-safe; two-twin routing) → sold at the **next close** (exactly ONE "
            "execution lag), loss classified ST (< 366 d) / LT, tax saving = loss × rate, "
            "**reinvested immediately** with the proceeds into the twin.\n"
            "* Costs: **2 bps one-way × traded NAV** on every leg (harvest sell, rebuy, terminal "
            "sale). No borrow (long-only, fully invested throughout).\n"
            "* Terminal: both books liquidate and pay per-lot CGT (LT if ≥ 366 d), losses credit "
            "symmetrically; `step_up=True` zeroes terminal tax on both books.\n"
            "* **Stated assumptions:** harvested losses fully usable against gains taxed at the "
            "ST/LT pair (the $3,000 ordinary-income cap never binds — bracketed by the 15/15 and "
            "0/0 rows); SPY↔IVV treated as *not substantially identical* (industry practice, "
            "never formally IRS-blessed).\n\n"
            "> 💡 **In plain words.** We rebuild a robo-advisor's daily harvesting bot with real "
            "tax-lot bookkeeping, run it over 33 years of S&P 500 history, and then run the *same "
            "market* through different tax brackets to see whose money the \"alpha\" really is."
        ),
        code(BOOT_CELL),

        md("## 0 · Data stamp + twin-fund equivalence\n\n"
           "The wash-sale-safe swap only keeps exposure unchanged if the twin really is the same "
           "exposure. SPY vs IVV daily returns:"),
        code(
            "if HAVE_REAL:\n"
            "    from quantlab import repro\n"
            "    print(repro.data_stamp('SPY total-return close', SPY.to_frame('SPY'),\n"
            "                           asof=data.AS_OF))\n"
            "    tc = data.twin_check(SPY, IVV)\n"
            "    print(f\"SPY vs IVV ({tc['n_days']:,} common days): corr {tc['corr']:.4f} | \"\n"
            "          f\"mean gap {tc['gap_ann_bps']:+.2f} bps/yr | \"\n"
            "          f\"tracking-diff vol {tc['te_ann_bps']:.0f} bps/yr\")\n"
            "else:\n"
            "    print('cache missing — see docs/results.md (fingerprint', R['fingerprint'], ')')\n"
        ),
        md(
            f"Correlation **{R['twin_corr']:.4f}** and a mean return gap of "
            f"**{R['twin_gap_bps']:+.2f} bps/yr** — economically the same asset (the "
            f"{R['twin_te_bps']} bps/yr daily tracking noise is close-timestamp jitter that nets "
            "out). We therefore run ONE price series and let the swap change only the tax lots — "
            "so the measured delta is **pure tax timing + costs**, uncontaminated by tracking "
            "luck.\n\n"
            "> 💡 **In plain words.** SPY and IVV are two wrappers around the same 500 stocks. "
            "Selling one and buying the other keeps your market ride identical — only the tax "
            "paperwork changes."
        ),

        md("## 1 · The full-period runs — lock-up punchline + the rate grid\n\n"
           "First the two 33-year investors: the 1993 **lump sum** (never harvests again after "
           "month one — SPY never revisits its 1993 basis) and the steady **contributor** "
           "($10k + $1k/month), whose fresh lots keep the machine fed. Then the same contributor "
           "tape through the rate grid."),
        code(
            "if HAVE_REAL:\n"
            "    rl = st.after_tax_alpha(SPY, st_rate=0.35, lt_rate=0.15)\n"
            "    print(f\"lump sum $100k, {rl['years']:.1f}y at 35/15: \"\n"
            "          f\"{rl['h']['n_harvests']} harvests, alpha {rl['alpha_bps']:+.1f} bps/yr\")\n"
            "    print()\n"
            "    for stt, ltt, su, lab in [(0.35,0.15,False,'35/15'), (0.24,0.15,False,'24/15'),\n"
            "                              (0.15,0.15,False,'15/15'), (0.35,0.15,True,'35/15+step-up'),\n"
            "                              (0.0,0.0,False,'0/0'), (0.0,0.0,True,'0/0+step-up')]:\n"
            "        rr = st.after_tax_alpha(SPY, initial=10_000, monthly_contrib=1_000,\n"
            "                                st_rate=stt, lt_rate=ltt, step_up=su)\n"
            "        print(f\"contributor {lab:<14s} alpha {rr['alpha_bps']:+6.1f} bps/yr | \"\n"
            "              f\"uplift {rr['uplift_pct']:+6.2f}% | harvests {rr['h']['n_harvests']}\")\n"
            "else:\n"
            "    for lab, a, u, n in R['grid']:\n"
            "        print(f'{lab:<36s} {a:+6.1f} bps/yr  {u:+6.2f}%  ({n} harvests)')\n"
        ),
        md(
            "Same 667 harvests in every contributor row — only the rates move the payoff:\n\n"
            + _grid_md_table() + "\n\n"
            f"* the ST/LT **rate arbitrage** (35→15) is worth ~{13.1 - 5.4:.1f} bps/yr on top of "
            f"the ~5.4 bps/yr pure-**deferral** value (the 15/15 row);\n"
            f"* **0/0 banks −0.4 bps/yr** — exactly the trading costs. The engine cannot "
            "manufacture value without a tax rate: the myth-check null on the *real* tape;\n"
            f"* **step-up at death** turns deferral into forgiveness — but only for brackets that "
            f"deduct something (+13.9 vs +13.1 at 35/15; still −0.4 at 0/0).\n\n"
            f"Detail at 35/15: harvested **${R['harvested_total']:,}** of losses "
            f"(**{R['st_share']:.1f}% short-term** — harvest resets the clock, so re-harvests are "
            f"almost always ST), tax savings reinvested **${R['tax_saved']:,}**, terminal tax "
            f"**${R['term_tax_tlh']:,}** (TLH) vs **${R['term_tax_bh']:,}** (B&H) — TLH *repays* "
            "part of its deductions at liquidation, as the arithmetic demands.\n\n"
            "> 💡 **In plain words.** The bot deducts at ~35 cents per dollar of loss today and "
            "repays at ~15 cents at the end — pocketing the 20-cent spread plus decades of "
            "interest-free use of the money. Set the rates to zero and the bot just burns "
            "commissions."
        ),

        md(f"## 2 · The Signal axis — {R['n_cohorts']} overlapping 10-year cohorts\n\n"
           "One lump-sum cohort per quarter-start (1993Q1 → 2016Q2 starts), each run 10 years at "
           "35/15. Overlap = 39 quarters, so the mean's *t* uses **Newey-West lag 40**."),
        code(
            "if HAVE_REAL:\n"
            "    co = st.cohort_alphas(SPY, horizon_years=10, st_rate=0.35, lt_rate=0.15)\n"
            "    a = co['alpha_bps'].to_numpy()\n"
            "    print(f'cohorts {len(co)} | mean {a.mean():+.1f} | median {np.median(a):+.1f} | '\n"
            "          f'sd {a.std(ddof=1):.1f} | min {a.min():+.1f} | max {a.max():+.1f} bps/yr')\n"
            "    print(f'HAC t (lag 40) = {st.nw_tstat(a, 40):+.2f} | share>0 '\n"
            "          f'{100*(a>0).mean():.1f}% | zero-harvest cohorts '\n"
            "          f\"{(co['n_harvests']==0).sum()}\")\n"
            "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "    ax1.plot(co.index, a, color=GREEN)\n"
            "    ax1.fill_between(co.index, 0, a, color=GREEN, alpha=.25)\n"
            "    ax1.axhline(0, color='k', lw=.8)\n"
            "    ax1.set_title('after-tax alpha by cohort start (bps/yr, 35/15)')\n"
            "    ax2.hist(a, bins=24, color=GREY)\n"
            "    ax2.axvline(a.mean(), color=GREEN, lw=2, label=f'mean {a.mean():+.0f}')\n"
            "    ax2.axvline(np.median(a), color=AMBER, lw=2, label=f'median {np.median(a):+.0f}')\n"
            "    ax2.legend(); ax2.set_title('cohort distribution')\n"
            "    plt.tight_layout()\n"
            "else:\n"
            "    print('cache missing — quoting R: mean', R['co_mean'], 'HAC t', R['hac_t'])\n"
        ),
        md(
            f"**Mean +{R['co_mean']:.1f} bps/yr, HAC *t* (lag {R['hac_lag']}) = "
            f"+{R['hac_t']:.2f}** — the mechanical after-tax delta clears the desk's *t* ≥ 2 bar "
            "on the real tape. But look at the *shape*: median **+"
            f"{R['co_median']:.1f}**, **{R['zero_harvest']} of {R['n_cohorts']}** cohorts never "
            f"harvest at all (start into a bull → basis locks immediately), max **+{R['co_max']:.0f}** "
            f"for cohorts that ride into 2000–02 or 2008. Cohorts starting **{R['best_start']}** "
            f"did best; **{R['worst_start']}** worst ({R['co_min']:+.1f} — a hair below zero: "
            "costs plus an unlucky ST/LT flip). By start decade: "
            f"{R['decade_means']}.\n\n"
            "**Cohort rate grid** (same sweep at other rate pairs — recomputed in "
            "`examples/verify.py`):\n\n"
            "| rates (ST/LT) | mean (bps/yr) | median | HAC *t* |\n|---|--:|--:|--:|\n"
            + "\n".join(f"| {lab} | **{m:+.1f}** | {md_:+.1f} | {t:+.2f} |"
                        for lab, m, md_, t in R["co_grid"]) +
            "\n\nSignificant at every rate pair — and the *magnitude* scales with the rate "
            "spread, as arithmetic must.\n\n"
            "> 💡 **In plain words.** Whether harvesting pays 0, 20 or 190 bps a year depends "
            "almost entirely on *when you started* — you need the market to hand you losses. "
            "The average is statistically solid; the typical experience is half the headline."
        ),

        md("## 3 · Path dependence + decay — where and when the harvest lives"),
        code(
            "if HAVE_REAL:\n"
            "    r35 = st.after_tax_alpha(SPY, initial=10_000, monthly_contrib=1_000,\n"
            "                             st_rate=0.35, lt_rate=0.15)\n"
            "    hy = st.harvest_yield_by_year(r35['h'])\n"
            "    bear = hy[hy.index.isin(R['bear_years'])].to_numpy()\n"
            "    calm = hy[~hy.index.isin(R['bear_years'])].to_numpy()\n"
            "    tw, dof = st.welch_t(bear, calm)\n"
            "    print(f'harvest yield: bear years {bear.mean():.2f}%/yr (n={len(bear)}) vs '\n"
            "          f'calm {calm.mean():.2f}%/yr (n={len(calm)}) | Welch t = {tw:+.2f} '\n"
            "          f'(dof {dof:.1f})')\n"
            "    rich = hy.sort_values(ascending=False).head(6)\n"
            "    print('richest years:', ', '.join(f'{int(y)}: {v:.2f}%' for y, v in rich.items()))\n"
            "else:\n"
            "    print('bear', R['bear_mean'], 'vs calm', R['calm_mean'], 'Welch t', R['welch_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "yrs = np.arange(1, 11)\n"
            "ax.bar(yrs, R['decay'], color=[GREEN]+[AMBER]*4+[RED]*5)\n"
            "ax.set_xticks(yrs); ax.set_xlabel('year since cohort inception')\n"
            "ax.set_ylabel('harvested loss, % of initial stake')\n"
            "ax.set_title('Basis lock-up: the harvest decays '\n"
            "             f\"{R['decay'][0]:.1f}% (y1) → {R['decay_tail']:.2f}%/yr (y7-10) — \"\n"
            "             'avg across 94 cohorts')\n"
        ),
        md(
            f"Bear years harvest **{R['bear_mean']:.2f}%** of NAV per year vs "
            f"**{R['calm_mean']:.2f}%** in calm years — **Welch *t* = +{R['welch_t']:.2f}** "
            f"(dof {R['welch_dof']}, n = 7 vs 27 calendar years). The harvest is a bear-market "
            "crop: 2008 (4.80%), 2002 (4.29%), 2009 (2.95%), 2001 (2.93%) dwarf everything since "
            "— even 2020 and 2022 yield ~0.1% because by then the contributor's NAV is dominated "
            "by old, deeply-appreciated lots (2022's drawdown never got below most of the "
            "2010s basis).\n\n"
            "And the decay curve is the industry's quiet footnote: **"
            f"{R['decay'][0]:.2f}%** of the stake harvested in year 1, "
            f"**{R['decay_tail']:.2f}%/yr** by years 7–10 — a ~40× fade. The 1993 lump sum "
            "harvests **zero** for 33 years. Without fresh cash, the machine retires itself.\n\n"
            "> 💡 **In plain words.** The bot only earns when prices sit below what *you* paid — "
            "which happens early, in crashes, or right after new deposits. A long-held account "
            "in a rising market gives it nothing to do."
        ),

        md("## 4 · Robustness — the harvest threshold\n\n"
           "Cohort sweep at other trigger levels (recomputed in `examples/verify.py`):\n\n"
           "| harvest when loss > | cohort mean (bps/yr) | HAC *t* | harvests/cohort |\n"
           "|---|--:|--:|--:|\n"
           + "\n".join(f"| {th:.1f}% | {m:+.1f} | {t:+.2f} | {n:.1f} |"
                       for th, m, t, n in R["thresholds"]) +
           "\n\nThe alpha is flat in the trigger (45.7–48.8 bps, *t* = 2.88–2.94): with 2 bps "
           "legs and a 35% deduction, almost any loss is worth harvesting, and the result is "
           "not an artefact of where we drew the 1% line."),

        md("## 5 · Synthetic control — the volatility knob (machinery proof)\n\n"
           "Deterministic GBM worlds, **20 seeds per cell** (single-seed baselines are banned "
           "on this desk). TLH alpha must rise with the planted volatility knob — volatility is "
           "what serves up harvestable drawdowns — and must sit at ~zero minus costs when rates "
           "are 0/0, whatever the path."),
        code(
            "ctl = st.synthetic_control(sigmas=(0.02, 0.18, 0.35), n_seeds=20)\n"
            "print(ctl.round(2))\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.errorbar(ctl.index, ctl['alpha_bps_35_15'], yerr=ctl['alpha_sd'],\n"
            "            fmt='o-', color=GREEN, capsize=4, label='alpha at 35/15')\n"
            "ax.plot(ctl.index, ctl['alpha_bps_rates0'], 's--', color=RED,\n"
            "        label='alpha at rates 0/0 (null)')\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_xlabel('planted GBM volatility (sigma)')\n"
            "ax.set_ylabel('after-tax alpha, bps/yr (mean of 20 seeds)')\n"
            "ax.set_title('Machinery proof: alpha rises with the planted vol knob; '\n"
            "             'zero rates -> zero alpha')\n"
            "ax.legend()\n"
        ),
        md(
            f"σ = 0.02 → **{R['syn'][0][1]:+.2f}** bps/yr (no drawdowns, nothing to harvest); "
            f"σ = 0.18 → **{R['syn'][1][1]:+.1f}**; σ = 0.35 → **{R['syn'][2][1]:+.1f}** — and "
            f"the 0/0-rate null sits at **{R['syn'][0][3]:+.2f} to {R['syn'][2][3]:+.2f}** "
            "bps/yr (the cost drag) at every σ. The engine detects exactly the effect it should "
            "and cannot manufacture value without a tax rate. *(Machinery proof only — never "
            "cited in support of the Signal stamp.)*"
        ),

        md(
            "## Verdict\n\n"
            f"* **Signal — REAL.** Mean after-tax alpha **+{R['co_mean']:.1f} bps/yr** across "
            f"{R['n_cohorts']} ten-year cohorts, **HAC *t* = +{R['hac_t']:.2f}** (lag "
            f"{R['hac_lag']}), robust to threshold (2.88–2.94) and rate pair (2.42–2.78); the "
            "0/0 run banks −0.4 bps/yr, proving the delta is pure tax arithmetic. The advertised "
            f"**0.3–1%/yr** holds only at the top bracket on the *mean*; the median cohort gets "
            f"+{R['co_median']:.1f} bps and the 33-year contributor +13.1.\n"
            "* **Tradability — FRAGILE.** Costs are trivial and capacity unbounded, but the "
            f"harvest is front-loaded ({R['decay'][0]:.1f}% → {R['decay_tail']:.2f}%/yr), "
            f"bear-fed (Welch *t* = +{R['welch_t']:.2f}), zero in {R['zero_harvest']}/"
            f"{R['n_cohorts']} cohorts, needs external ST gains for the headline rate — and a "
            "0.25%/yr robo fee exceeds the median cohort's take.\n"
            "* **Who gets ZERO? — CONFIRMED.** The 0%-bracket retiree/donor: **−0.4 bps/yr** "
            "(pure costs), with or without step-up. The alpha is a private rate arbitrage — it "
            "is *your bracket*, monetised.\n\n"
            "*Reproduce every number: `python examples/verify.py` · frozen copy: "
            "[docs/results.md](../docs/results.md) · literature: "
            "[docs/references.md](../docs/references.md). Not investment or tax advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    return nb


if __name__ == "__main__":
    for name, builder in [("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)]:
        path = os.path.join(HERE, name)
        nbf.write(builder(), path)
        print("wrote", path)
