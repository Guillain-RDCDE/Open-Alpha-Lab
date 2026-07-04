"""Generate the two narrative notebooks for Study 619 (BITO Roll Drag).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached yfinance tape
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic planted-basis control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return closes,
# BITO 2021-10-20 -> 2026-06-30, IBIT era 2024-01-11+, fingerprint 48e3e370f939).
R = dict(
    start="2021-10-20", end="2026-06-30", asof="2026-06-30", years=4.69,
    n_days=1176, n_months=56, fingerprint="48e3e370f939",
    # BITO vs spot
    spot_drag_ann=-4.25, spot_gap_bps_day=-1.69, spot_hac_t=-1.67,
    spot_tr_bito=-31.45, spot_tr_spot=-11.26, spot_shortfall_pp=-20.18, wealth_ratio=0.773,
    mo_gap_bps=-34.49, mo_hac_t=-2.32, mo_neg_share=69.6,
    beta_ibit=1.003, beta_spot=0.947,
    # BITO vs IBIT
    ibit_start="2024-01-11", ibit_days=617, ibit_years=2.47,
    ibit_drag_ann=-5.14, ibit_gap_bps_day=-2.04, ibit_hac_t=-7.68,
    ibit_hac_t_lag5=-7.22, ibit_hac_t_lag63=-7.27,
    ibit_tr_bito=9.99, ibit_tr_ibit=25.01, ibit_shortfall_pp=-15.02, ibit_wealth_ratio=0.880,
    ctrl_ibit_spot_ann=0.28, ctrl_ibit_spot_t=0.10,
    fee_bito=0.95, fee_ibit=0.25, fee_gap=0.70, carry_beyond_fees=4.44,
    # roll-window attribution: (race, in bps/d, n_in, out bps/d, n_out, welch t, day%, shortfall%)
    roll=[("vs spot", -4.14, 285, -0.91, 891, -0.33, 24.2, 54.6),
          ("vs IBIT", -2.84, 150, -1.78, 467, -0.87, 24.3, 33.6)],
    # basis
    contango_share=58.5, median_basis_bps=19.1, mean_days_to_exp=13.9,
    ann_basis_median=4.90, ann_basis_2022=-4.72, ann_basis_2426=7.16,
    # calendar years: (year, bps/mo, n months, cum pp)
    years_tbl=[(2021, -163.2, 2, -3.24), (2022, 33.3, 12, 3.63), (2023, -56.4, 12, -6.60),
               (2024, -56.9, 12, -6.78), (2025, -43.7, 12, -5.14), (2026, -20.0, 6, -1.20)],
    regime_2022=33.3, regime_2023p=-47.7, regime_welch_t=0.98,
    daily_split_contango=-6.16, daily_split_backward=-3.35, daily_split_t=-1.09,
    # spread: (borrow %, gross, net, hac t, sharpe)
    spread=[(0.0, 5.14, 4.90, 7.34, 2.55), (2.0, 5.14, 2.90, 4.34, 1.51),
            (5.0, 5.14, -0.10, -0.16, -0.05)],
    spread_vol=1.92, spread_worst_day=-0.56,
    # synthetic: (planted %/yr, measured, hac t, roll welch t)
    syn=[(0.00, 0.22, 0.70, 0.20), (-10.95, -10.42, -32.55, 0.11)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Reason_to_survive%3F: Busted](https://img.shields.io/badge/Reason_to_survive%3F-Busted-8b949e?style=flat-square)\n\n"
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

from bito_roll_drag import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("real tape cached:", HAVE_REAL,
      "| rows:", (0 if F is None else len(F)),
      "| window:", ("-" if F is None else f"{F.index.min().date()} -> {F.index.max().date()}"))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The bitcoin ETF that pays a toll every month 🎢\n"
            "### BITO vs actual bitcoin — where the missing 20 percentage points went, in plain English\n\n"
            + BADGES +
            "In October 2021 the US finally got a bitcoin ETF — with an asterisk. **BITO** wasn't "
            "allowed to hold bitcoin. It holds **futures**: contracts to buy bitcoin *next month*, "
            "listed on a Chicago exchange. Every month those contracts expire, and the fund sells "
            "them and buys the next month's batch. That monthly shuffle is called **the roll**.\n\n"
            "The folklore says the roll is a **toll booth**: next month's bitcoin usually costs a "
            "little *more* than today's (a premium called **contango**), the fund keeps buying at a "
            "premium that then melts away, and the holder pays for it — forever.\n\n"
            "Since January 2024 there's a perfect way to check: **IBIT** and its siblings hold "
            "*actual* bitcoin. Same animal, no futures, no roll. We race all three: BITO, spot "
            "bitcoin, and IBIT — counting **every distribution BITO ever paid** (that part matters, "
            "you'll see).\n\n"
            "> 📓 Want the *t*-stats, the roll-window attribution and the borrow math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does BITO lag actual bitcoin? | **Yes.** Since inception a dollar in BITO is worth about **{R['wealth_ratio']*100:.0f} cents** of the spot-bitcoin dollar — **{abs(R['spot_shortfall_pp']):.0f} points** of total return went missing in {R['years']:.1f} years. |\n"
            f"| Is it luck, or a toll? | A toll. Against the spot ETF (same closing bell, cleanest possible ruler) BITO loses **{abs(R['ibit_drag_ann']):.1f}% per year** — and the quants' test says that's about as far from luck as it gets (*t* = {R['ibit_hac_t']:.1f}). |\n"
            f"| Is it the fee? | Only partly. Fees explain **{R['fee_gap']:.1f}pt**; the other **~{R['carry_beyond_fees']:.1f}%/yr** is the futures premium melting, month after month. |\n"
            f"| Is the toll booth at the roll? | **No — that's the twist.** The toll accrues a little *every day* as the premium melts; the roll week itself shows no special bleed. |\n"
            f"| Did it ever flip? | **Yes: 2022.** In the bear market the premium inverted and BITO actually *beat* spot by ~{R['years_tbl'][1][3]:.1f}pt that year. Contango giveth back, occasionally. |\n"
            f"| Any reason to hold BITO today? | For bitcoin *exposure*, none we can find: same risk (beta ≈ 1.00 on IBIT), {abs(R['ibit_drag_ann']):.1f}%/yr lighter. The famous \"monthly income\" is your own money handed back. |"
        ),

        md(
            "## Act I — three ways to buy the same coin\n\n"
            "Below: a dollar into **spot bitcoin**, into **BITO** (with every monthly distribution "
            "reinvested — the fair way to count), and into **IBIT** from the day it listed. Same "
            "underlying, one of them pays a toll."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots()\n"
            "    w = F[['spot', 'bito']] / F[['spot', 'bito']].iloc[0]\n"
            "    ax.plot(w.index, w['spot'], color=GREY, lw=2, label='spot bitcoin (BTC-USD)')\n"
            "    ax.plot(w.index, w['bito'], color=RED, lw=2, label='BITO (total return, distributions reinvested)')\n"
            "    ib = F['ibit'].dropna(); ib = ib / ib.iloc[0] * w['spot'].loc[ib.index[0]]\n"
            "    ax.plot(ib.index, ib, color=GREEN, lw=2, label='IBIT (spot ETF, from Jan-2024, rebased)')\n"
            "    ax.set_title('Growth of $1 — the futures ETF vs the real thing')\n"
            "    ax.set_ylabel('wealth (log scale)'); ax.set_yscale('log'); ax.legend()\n"
            "    plt.show()\n"
            "    print(f\"final: spot {w['spot'].iloc[-1]:.3f}  BITO {w['bito'].iloc[-1]:.3f}  \"\n"
            "          f\"-> BITO/spot wealth ratio {w['bito'].iloc[-1]/w['spot'].iloc[-1]:.3f}\")\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', R['spot_tr_bito'], '% BITO vs', R['spot_tr_spot'], '% spot')"
        ),
        md(
            f"Same coin, different vehicles. Spot bitcoin finished the window at **{R['spot_tr_spot']:+.1f}%**; "
            f"BITO — *with every distribution reinvested* — at **{R['spot_tr_bito']:+.1f}%**. The gap "
            f"(**{R['spot_shortfall_pp']:.1f} points**) is the toll.\n\n"
            "> ⚠️ **About those distributions.** BITO is famous for fat monthly payouts. They are not "
            "extra return — they're the fund handing your own gains back (and its price drops by the "
            "same amount). Anyone comparing BITO's *price* chart to bitcoin is overstating the toll "
            "enormously; we reinvest everything and the toll is still there."
        ),

        md(
            "## Act II — watch the toll accrue\n\n"
            "Divide BITO's wealth by spot's wealth: flat means fair tracking, downhill means the "
            "toll is running. One year stands out."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w = F[['spot', 'bito']] / F[['spot', 'bito']].iloc[0]\n"
            "    ratio = w['bito'] / w['spot']\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(ratio.index, ratio, color=RED, lw=2)\n"
            "    ax.axhline(1.0, color=GREY, ls='--', lw=1)\n"
            "    ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'), color=GREEN, alpha=.12)\n"
            "    ax.annotate('2022: curve inverted,\\nBITO briefly WINS', xy=(pd.Timestamp('2022-07-01'), ratio.loc['2022'].max()),\n"
            "                ha='center', fontsize=9, color=GREEN)\n"
            "    ax.set_title('BITO wealth / spot wealth — the toll meter (down = paying)')\n"
            "    ax.set_ylabel('ratio')\n"
            "    plt.show()\n"
            "else:\n"
            "    print('cache missing - frozen wealth ratio:', R['wealth_ratio'])"
        ),
        md(
            "The slope is the story. Downhill in 2021, 2023, 2024, 2025, 2026 — every year the "
            "futures curve sat in **contango** (next month more expensive than today). And one "
            "green stretch: **2022**, the bear market, when fear inverted the curve "
            "(**backwardation** — next month *cheaper*) and the roll briefly *paid* BITO holders "
            f"about **{R['years_tbl'][1][3]:+.1f} points**. The toll is not a law of nature; it's the price "
            "of the crowd usually leaning bullish.\n\n"
            "| year | toll (per month) | over the year |\n|---|--:|--:|\n"
            + "\n".join(f"| {y} | {b:+.0f} bps | {c:+.2f} pp |" for y, b, n, c in R["years_tbl"])
            + "\n\n> 🔬 **For the quants:** the monthly toll vs spot averages "
            f"**{R['mo_gap_bps']:.0f} bps/month** (HAC *t* = {R['mo_hac_t']:.2f}, {R['mo_neg_share']:.0f}% "
            "of months negative); the matched-close race vs IBIT reads "
            f"**{R['ibit_drag_ann']:.1f}%/yr at *t* = {R['ibit_hac_t']:.1f}**."
        ),

        md(
            "## Act III — the twist: the toll booth isn't where you think\n\n"
            "\"Pays a toll every month it **rolls**\" — so the damage should show up in roll week, "
            "right? We flagged the five trading days into each futures expiry (last Friday of the "
            "month) and compared the bleed inside vs outside that window.\n\n"
            f"Inside: **{R['roll'][1][1]:.1f} bps/day**. Outside: **{R['roll'][1][3]:.1f} bps/day** "
            f"(vs IBIT). Statistically indistinguishable (*t* ≈ {R['roll'][1][5]:.1f}).\n\n"
            "That's not a failed test — it's the mechanism showing its true face. The fund buys "
            "next month's bitcoin at a premium, and that premium **melts a little every single "
            "day** as the contract drifts toward reality. The roll itself just resets the meter. "
            "The folklore is right about the *size* of the toll and wrong about the *where*."
        ),

        md(
            "## Act IV — so… why does BITO still exist?\n\n"
            "In 2021, BITO was the only game in town — US investors couldn't buy a spot bitcoin "
            "fund at any price, and BITO gathered a billion dollars in two days. Fair enough.\n\n"
            "Since **January 2024** the spot ETFs exist. Same exposure (the quants measure beta "
            f"**{R['beta_ibit']:.2f}** — identical), **{abs(R['ibit_drag_ann']):.1f}%/yr cheaper** in "
            "realized returns. The classic defenses:\n\n"
            "- **\"But the monthly income!\"** — it's your own capital, repackaged. We reinvested "
            "every cent of it and BITO *still* lost by 15 points in 2½ years.\n"
            "- **\"But options!\"** — was true; IBIT options listed in **Nov-2024** and are now the "
            "deeper market.\n"
            "- **\"But my retirement account!\"** — spot ETFs sit in IRAs just fine now.\n\n"
            "What's left is habit and habitat. For bitcoin *exposure*, the tape finds **no reason** "
            "— stamped **Busted** on the grey axis.\n\n"
            "---\n\n"
            "## The verdict\n\n"
            f"**Real** — the toll exists and it's decisive: **{R['ibit_drag_ann']:.1f}%/yr** against the "
            f"spot ETF (*t* = {R['ibit_hac_t']:.1f}), **{R['wealth_ratio']*100:.0f} cents on the dollar** "
            "since inception vs spot. **Fragile** as a trade — shorting BITO against IBIT nets a "
            "thin ~3%/yr that dies if borrow gets expensive, and the toll itself flipped sign in "
            "2022. And **Busted** on the myth that BITO still earns its keep.\n\n"
            "*Full numbers: [docs/results.md](../docs/results.md) · stats: "
            "[02_for_the_quants.ipynb](02_for_the_quants.ipynb) · siblings in the mechanical-decay "
            "family: [61-slow-burn](../../61-slow-burn/), [100-melting-ice](../../100-melting-ice/), "
            "[375-vxx-roll-decay](../../375-vxx-roll-decay/). Research & education, not investment "
            "advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# BITO Roll Drag — the quant teardown 🎢\n\n"
            + BADGES +
            "**Claim under test:** the futures-based bitcoin ETF pays a toll every month it rolls — "
            "measurable daylight between BITO and spot.\n\n"
            "**Design.** Total-return legs throughout (`auto_adjust`; BITO's large monthly "
            "distributions make price-only comparisons dishonest). Three races: BITO vs spot "
            "(inception 2021-10-20 →), BITO vs IBIT matched-close (2024-01-11 →), IBIT vs spot "
            "(timestamp-bias control). Inference: Newey-West HAC *t* on the mean gap; Welch *t* for "
            "roll-window and regime splits; the single execution lag is the prior-day basis sign in "
            "the regime split. As-of **2026-06-30**, fingerprint `48e3e370f939`.\n\n"
            "> 💡 **In plain words:** we count every payout BITO ever made, then ask whether what's "
            "left still lags the real coin — and whether the lag is luck or arithmetic."
        ),
        code(BOOT_CELL),

        md("## 0 · Data stamp — pin the tape before quoting it"),
        code(
            "from quantlab import repro\n"
            "if HAVE_REAL:\n"
            "    print(repro.data_stamp('BITO/IBIT/BTC-USD/BTC=F daily TR closes', F, asof=data.ASOF))\n"
            "    print('complete months:', R['n_months'], '| IBIT era days:', R['ibit_days'])\n"
            "else:\n"
            "    print('cache missing - frozen fingerprint:', R['fingerprint'])"
        ),

        md(
            "## 1 · Headline — the drag on both rulers\n\n"
            "Vs spot, the *daily* gap is noisy by construction (Yahoo's BTC-USD closes ~00:00 UTC, "
            "BITO 16:00 ET) — it attenuates the daily beta to "
            f"**{R['beta_spot']:.3f}** and inflates the daily HAC SE. The honest vs-spot statistic is "
            "**monthly**, where the offset washes out. The matched-close IBIT race has no such "
            "problem (beta "
            f"**{R['beta_ibit']:.3f}**), and the IBIT-vs-spot control shows the offset adds no *bias*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s  = st.summarize_drag(F, 'bito', 'spot')\n"
            "    m  = st.monthly_gap(F, 'bito', 'spot')\n"
            "    si = st.summarize_drag(F, 'bito', 'ibit')\n"
            "    sc = st.summarize_drag(F, 'ibit', 'spot')\n"
            "    print(f\"BITO vs spot : {s['drag_ann_pct']:+.2f}%/yr  HAC(21) t={s['hac_t']:+.2f}  \"\n"
            "          f\"(daily, timestamp-noisy) | shortfall {s['shortfall_pp']:+.2f} pp, wealth ratio {s['wealth_ratio']:.3f}\")\n"
            "    print(f\"   monthly   : {m['gap_bps_month']:+.2f} bps/mo  HAC(3) t={m['hac_t']:+.2f}  \"\n"
            "          f\"({m['neg_share']*100:.1f}% of {m['n_months']} months negative)\")\n"
            "    print(f\"BITO vs IBIT : {si['drag_ann_pct']:+.2f}%/yr  HAC(21) t={si['hac_t']:+.2f}  \"\n"
            "          f\"| shortfall {si['shortfall_pp']:+.2f} pp in {si['years']:.2f} yr\")\n"
            "    for L in (5, 63):\n"
            "        print(f\"   HAC({L:>2}) t = {st.hac_t(st.daily_gap(F,'bito','ibit').values, lags=L)['t']:+.2f}\")\n"
            "    print(f\"IBIT vs spot : {sc['drag_ann_pct']:+.2f}%/yr  t={sc['hac_t']:+.2f}   <- control: no timestamp bias in means\")\n"
            "    print(f\"fee arithmetic: {R['fee_bito']}% - {R['fee_ibit']}% = {R['fee_gap']}%/yr fees; \"\n"
            "          f\"observed {si['drag_ann_pct']:+.2f}%/yr -> ~{abs(si['drag_ann_pct'])-R['fee_gap']:.2f}%/yr carry toll\")\n"
            "else:\n"
            "    print('frozen:', R['ibit_drag_ann'], '%/yr vs IBIT, HAC t', R['ibit_hac_t'])"
        ),
        md(
            f"> 💡 **In plain words:** vs the spot ETF — same closing bell, same coin — BITO loses "
            f"**{abs(R['ibit_drag_ann']):.1f}% a year** and the odds that's luck are astronomically small "
            f"(*t* = {R['ibit_hac_t']:.1f}). Fees explain {R['fee_gap']:.1f}pt; the melting futures premium "
            f"explains the other ~{R['carry_beyond_fees']:.1f}."
        ),

        md("## 2 · The toll meter — cumulative wealth ratios"),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots()\n"
            "    w = F[['spot','bito']]/F[['spot','bito']].iloc[0]\n"
            "    ax.plot(w.index, w['bito']/w['spot'], color=RED, lw=2, label='BITO / spot (TR)')\n"
            "    sub = F[['bito','ibit']].dropna(); sub = sub/sub.iloc[0]\n"
            "    ax.plot(sub.index, sub['bito']/sub['ibit'], color=AMBER, lw=2, label='BITO / IBIT (TR, matched closes)')\n"
            "    ax.axhline(1, color=GREY, ls='--', lw=1)\n"
            "    ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'), color=GREEN, alpha=.10)\n"
            "    ax.set_title('Toll meters: futures-ETF wealth / benchmark wealth (2022 backwardation shaded)')\n"
            "    ax.set_ylabel('ratio'); ax.legend()\n"
            "    plt.show()\n"
            "else:\n"
            "    print('frozen wealth ratios:', R['wealth_ratio'], R['ibit_wealth_ratio'])"
        ),

        md(
            "## 3 · Roll-window attribution — *where* is the toll paid?\n\n"
            "Window = 5 trading days ending on each month's expiry Friday (CME BTC futures "
            "terminate the last Friday; BITO rolls into expiry). If the toll were an execution "
            "event, the window would carry the shortfall; if it's carry, the bleed is diffuse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    flags = data.roll_window_flags(F.index, width=5)\n"
            "    for ref in ('spot', 'ibit'):\n"
            "        ra = st.roll_attribution(F, flags, 'bito', ref)\n"
            "        print(f\"vs {ref:4}: in {ra['in_bps_day']:+.2f} bps/d (n={ra['n_in']}) | out \"\n"
            "              f\"{ra['out_bps_day']:+.2f} (n={ra['n_out']}) | Welch t {ra['welch_t']:+.2f} | \"\n"
            "              f\"days {ra['share_days']*100:.1f}% -> shortfall {ra['share_shortfall']*100:.1f}%\")\n"
            "    fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "    ra = st.roll_attribution(F, flags, 'bito', 'ibit')\n"
            "    ax.bar(['roll window','other days'], [ra['in_bps_day'], ra['out_bps_day']], color=[AMBER, GREY])\n"
            "    ax.set_ylabel('mean daily gap (bps/day)')\n"
            "    ax.set_title(f\"BITO-IBIT daily gap in vs out of roll week (Welch t = {ra['welch_t']:+.2f})\")\n"
            "    plt.show()\n"
            "else:\n"
            "    for row in R['roll']: print(row)"
        ),
        md(
            "> 💡 **In plain words:** no toll booth at the roll. The premium melts a little every "
            "day; the roll only resets which contract is melting. The folklore's *size* is right, "
            "its *location* is wrong — Welch *t* = "
            f"{R['roll'][0][5]:.2f} (vs spot) / {R['roll'][1][5]:.2f} (vs IBIT), nowhere near significance."
        ),

        md(
            "## 4 · Contango, backwardation, and the sign of the toll\n\n"
            "Front basis = `BTC=F / BTC-USD − 1`. Median **+"
            f"{R['median_basis_bps']:.1f} bps** at a mean **{R['mean_days_to_exp']:.1f}** days to expiry — "
            f"**{R['ann_basis_median']:+.2f}%/yr annualized**, which is the carry the roll locks in, and it "
            f"matches the measured beyond-fee drag (~{R['carry_beyond_fees']:.2f}%/yr) almost exactly."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.basis_series(F)\n"
            "    tau = []\n"
            "    for d in b.index:\n"
            "        lf = data.last_friday(d.year, d.month)\n"
            "        if d > lf:\n"
            "            nm = d + pd.offsets.MonthBegin(1); lf = data.last_friday(nm.year, nm.month)\n"
            "        tau.append(max((lf - d).days, 1))\n"
            "    ann_b = pd.Series(b.values/np.array(tau)*365.0, index=b.index)\n"
            "    print(f'contango share of days: {(b>0).mean()*100:.1f}% | median ann basis {ann_b.median()*100:+.2f}%/yr')\n"
            "    print(f'2022 median ann basis: {ann_b[ann_b.index.year==2022].median()*100:+.2f}%/yr | '\n"
            "          f'2024-26: {ann_b[ann_b.index.year>=2024].median()*100:+.2f}%/yr')\n"
            "    m = st.monthly_gap(F, 'bito', 'spot'); gap = m['series']\n"
            "    back, cont = gap[gap.index.year==2022], gap[gap.index.year>=2023]\n"
            "    w = st.welch_t(back.values, cont.values)\n"
            "    print(f'2022 gap {back.mean()*1e4:+.1f} bps/mo vs 2023+ {cont.mean()*1e4:+.1f} bps/mo -> Welch t {w[\"t\"]:+.2f}')\n"
            "    fig, ax = plt.subplots()\n"
            "    q = ann_b.rolling(21, min_periods=10).median()\n"
            "    ax.plot(q.index, q*100, color=GREY, lw=1.5, label='21d rolling median annualized basis')\n"
            "    ax.axhline(0, color=RED, ls='--', lw=1)\n"
            "    ax.fill_between(q.index, 0, q*100, where=(q>0), color=AMBER, alpha=.3, label='contango (toll runs)')\n"
            "    ax.fill_between(q.index, 0, q*100, where=(q<=0), color=GREEN, alpha=.3, label='backwardation (toll flips)')\n"
            "    ax.set_ylabel('%/yr'); ax.set_title('The carry BITO rolls into'); ax.legend()\n"
            "    plt.show()\n"
            "else:\n"
            "    print('frozen: contango share', R['contango_share'], '% | ann basis median', R['ann_basis_median'])"
        ),
        md(
            "**Attribution honesty.** The year-level contrast is *directionally* the carry story — "
            f"2022 (median ann basis {R['ann_basis_2022']:+.1f}%) saw the gap flip to "
            f"**{R['regime_2022']:+.1f} bps/mo** vs **{R['regime_2023p']:+.1f}** in the contango era — but on "
            f"12-vs-42 noisy months it reads Welch *t* = **{R['regime_welch_t']:+.2f}**: *suggestive, not "
            "certified*. The daily lag-1 basis-sign split (vs IBIT: contango "
            f"{R['daily_split_contango']:+.2f}%/yr vs backwardation {R['daily_split_backward']:+.2f}%/yr, "
            f"Welch *t* = {R['daily_split_t']:+.2f}) is underpowered — and §6 shows that *kind* of split "
            "is artifact-prone, so we do not lean on it.\n\n"
            "> 💡 **In plain words:** we can prove the toll; proving *statistically* that it runs only "
            "when the curve points up needs more backwardation years than bitcoin has given us. The "
            "one it did give (2022) behaved exactly as the story predicts."
        ),

        md(
            "## 5 · Tradability — harvesting the toll\n\n"
            "Long IBIT / short BITO, dollar-neutral, daily rebalance, IBIT era only (before 2024 "
            "there was no clean long leg). Short pays borrow on full notional; 2 bps one-way on "
            "rebalancing turnover; positions from the prior close."
        ),
        code(
            "if HAVE_REAL:\n"
            "    curves = {}\n"
            "    for borrow in (0.0, 0.02, 0.05):\n"
            "        sp = st.spread_trade(F, borrow_ann=borrow, cost_bps=2.0)\n"
            "        print(f\"borrow {borrow*100:>4.1f}%: gross {sp['gross_ann_pct']:+.2f}%/yr -> net \"\n"
            "              f\"{sp['net_ann_pct']:+.2f}%/yr  HAC t {sp['net_hac_t']:+.2f}  Sharpe {sp['sharpe']:+.2f}\")\n"
            "    sub = F[['ibit','bito']].dropna().pct_change().dropna()\n"
            "    for borrow, colr in ((0.0, GREEN), (0.02, AMBER), (0.05, RED)):\n"
            "        net = (sub['ibit']-sub['bito']) - borrow/252 - 2e-4*(sub['ibit'].abs()+sub['bito'].abs())\n"
            "        curves[borrow] = (1+net).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    for borrow, colr in ((0.0, GREEN), (0.02, AMBER), (0.05, RED)):\n"
            "        ax.plot(curves[borrow].index, curves[borrow], color=colr, lw=2, label=f'borrow {borrow*100:.0f}%/yr')\n"
            "    ax.axhline(1, color=GREY, ls='--', lw=1)\n"
            "    ax.set_title('Long IBIT / short BITO, net equity curve by borrow cost')\n"
            "    ax.set_ylabel('growth of $1'); ax.legend()\n"
            "    plt.show()\n"
            "else:\n"
            "    for row in R['spread']: print(row)"
        ),
        md(
            f"> 💡 **In plain words:** at cheap borrow this is a genuine, low-vol carry "
            f"(**+{R['spread'][1][2]:.1f}%/yr net, Sharpe {R['spread'][1][4]:.1f}**, vol {R['spread_vol']:.1f}%, "
            f"worst day {R['spread_worst_day']:.2f}%). At 5% borrow it's dead ({R['spread'][2][2]:+.1f}%/yr). "
            "A ~3% absolute carry that lives or dies on the stock-loan desk, on a couple of billion "
            "of float, is **Fragile** — and the risk-free version (just *hold the spot ETF*) is cost "
            "avoidance, not alpha."
        ),

        md(
            "## 6 · Synthetic control — planted basis, faithful machinery *(never market evidence)*\n\n"
            "Deterministic GBM spot + front-month curve `F = S·exp(b·τ)` with `b = basis_ann + "
            "AR(1) noise`; the futures-ETF NAV rolls at each expiry and pays a fee. Planted drag = "
            "basis + fee. The null (basis 0, fee 0) must stay silent."
        ),
        code(
            "for basis, fee in ((0.0, 0.0), (0.10, 0.0095)):\n"
            "    w = data.synthetic_world(basis_ann=basis, fee_ann=fee, seed=619)\n"
            "    s = st.summarize_drag(w)\n"
            "    ra = st.roll_attribution(w, data.roll_window_flags(w.index))\n"
            "    print(f'planted {-(basis+fee)*100:+6.2f}%/yr: measured {s[\"drag_ann_pct\"]:+6.2f}%/yr  '\n"
            "          f'HAC t {s[\"hac_t\"]:+7.2f}  roll Welch t {ra[\"welch_t\"]:+.2f} (diffuse by construction)')"
        ),
        md(
            "The detector recovers a planted toll (−10.42 vs −10.95 planted; the residual is "
            "calendar-vs-trading-day annualization) and invents nothing from basis noise (null "
            "*t* = +0.70). The planted drag is **diffuse across the month by construction** — "
            "matching the real tape's roll-window verdict in §3.\n\n"
            "**A trap the control exposed:** conditioning the *daily* gap on the lagged basis sign "
            "manufactures a large spurious split from mean-reverting basis noise alone (the noise "
            "term `τ·Δb` reverses after extreme basis days). That is why §4's regime attribution "
            "leans on calendar-year basis levels, not the daily sign flip."
        ),

        md(
            "## 7 · Verdict\n\n"
            f"- **Signal — REAL.** The toll is on the tape twice over: **{R['mo_gap_bps']:.1f} bps/month** "
            f"vs spot (HAC *t* = {R['mo_hac_t']:.2f}, {R['mo_neg_share']:.0f}% of months negative, "
            f"{R['spot_shortfall_pp']:.1f} pp cumulative) and **{R['ibit_drag_ann']:.2f}%/yr** vs IBIT on "
            f"matched closes (HAC(21) *t* = {R['ibit_hac_t']:.2f}, lag-robust). Fee gap {R['fee_gap']:.2f}% + "
            f"carry ~{R['carry_beyond_fees']:.2f}% ≈ the {R['ann_basis_median']:+.2f}%/yr median annualized "
            "front basis: the arithmetic closes. No survivorship (single live instrument).\n"
            f"- **Tradability — FRAGILE.** The active harvest nets **+{R['spread'][1][2]:.2f}%/yr "
            f"(HAC *t* = +{R['spread'][1][3]:.2f}, Sharpe {R['spread'][1][4]:.2f})** at 2% borrow and dies at "
            "~5%; thin, borrow-gated, capacity-bounded. The toll flipped sign in the 2022 "
            "backwardation. The passive fix is substitution, not alpha.\n"
            f"- **\"Any reason BITO survives?\" — BUSTED.** Beta on IBIT = {R['beta_ibit']:.3f} — identical "
            f"exposure — at {R['ibit_drag_ann']:.2f}%/yr after crediting every distribution; the access and "
            "options rationales migrated to the spot ETFs by Nov-2024. Habitat, not performance.\n\n"
            "*Mirrors [docs/results.md](../docs/results.md); numbers frozen in `R` "
            "(fingerprint `48e3e370f939`). Siblings: [61-slow-burn](../../61-slow-burn/), "
            "[100-melting-ice](../../100-melting-ice/), [375-vxx-roll-decay](../../375-vxx-roll-decay/).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    return nb


if __name__ == "__main__":
    for name, builder in (("01_for_the_curious.ipynb", build_curious),
                          ("02_for_the_quants.ipynb", build_quants)):
        nb = builder()
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path)
