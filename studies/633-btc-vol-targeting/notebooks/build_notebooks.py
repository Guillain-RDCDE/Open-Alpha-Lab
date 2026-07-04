"""Generate the two narrative notebooks for Study 633 (BTC Vol Targeting).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached BTC-USD
tape under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance BTC-USD,
# 2014-09-17 -> 2026-06-30, 4,305 daily bars, as-of 2026-06-30, fp 9529d5277775).
R = dict(
    start="2014-09-17", end="2026-06-30", years=11.8, n_days=4305, n_race=4274,
    target=30, window=30, cap=1.5,
    s_cagr=38.12, s_vol=35.54, s_sharpe=1.087, s_dd=-52.90, s_wealth=43.9,
    b_cagr=53.63, b_vol=66.50, b_sharpe=0.981, b_dd=-83.40, b_wealth=152.6,
    avg_w=0.604, share_lev=10.1, share_cap=1.1, turnover=6.97,
    gap_ann=-10.64, t_gap=-1.00,
    alpha=7.18, t_alpha=1.42, beta=0.482, appraisal=0.468,
    # vol thermostat
    vt_median=31.0, vt_p10=22.4, vt_p90=47.0, vt_band=72.8,
    vt_bh_median=54.4, vt_bh_p90=95.1,
    # crash windows: (name, strat DD %, B&H DD %)
    crashes=[("2018 bear", -52.61, -83.40), ("COVID 2020", -31.95, -51.86),
             ("2021-22 bear", -52.86, -76.63), ("full sample", -52.90, -83.40)],
    # grid: (target %, window d, sharpe, maxDD %, cagr %, alpha t, growth-gap t)
    grid=[(20, 20, 1.018, -40.56, 24.94, 0.91, -1.66),
          (20, 30, 1.095, -38.61, 26.16, 1.48, -1.49),
          (20, 60, 1.035, -36.57, 22.95, 0.99, -1.66),
          (30, 20, 1.018, -53.78, 35.93, 0.90, -1.27),
          (30, 30, 1.087, -52.90, 38.12, 1.42, -1.00),
          (30, 60, 1.034, -50.38, 33.71, 0.98, -1.31),
          (50, 20, 0.971, -75.36, 47.57, 0.44, -0.69),
          (50, 30, 1.013, -74.01, 50.58, 0.85, -0.27),
          (50, 60, 1.004, -70.47, 48.91, 0.71, -0.47)],
    # costs: (one-way bps, borrow %, sharpe, cagr %, maxDD %, alpha %/yr, alpha t)
    costs=[(0, 0, 1.087, 38.12, -52.90, 7.18, 1.42),
           (5, 0, 1.077, 37.64, -53.04, 6.83, 1.35),
           (10, 2, 1.066, 37.10, -53.19, 6.44, 1.27),
           (20, 5, 1.045, 36.07, -53.49, 5.68, 1.13)],
    # placebo (200 shuffled-vol seeds)
    pl_mean_alpha=0.23, pl_mean_t=0.06, pl_sd_t=0.97, pl_p_alpha=0.105,
    pl_mean_dd=-67.34, pl_p_dd=0.010, pl_seeds=200,
    # third axis
    giveup=15.51, kept_cagr=71.1, kept_wealth=28.8,
    # synthetic control (20 seeds/world)
    syn_null_alpha=0.21, syn_null_t=0.02, syn_null_share=0,
    syn_pl_alpha=27.02, syn_pl_t=4.33, syn_pl_share=100,
    fingerprint="9529d5277775",
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Same_ride%3F: Busted](https://img.shields.io/badge/Same_ride%3F-Busted-8b949e?style=flat-square)\n\n"
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

from btc_vol_targeting import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    RET = data.daily_returns()                      # sliced to the study as-of
    OV = st.run_overlay(RET)                        # headline 30% / 30d / 1.5x, gross
else:
    RET = OV = None
print("real BTC cache present:", HAVE_REAL,
      "| days:", (0 if RET is None else len(RET)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can a thermostat tame Bitcoin? 🌡️\n"
            "### Vol targeting BTC at 30% — half the heart attacks, but *not* the same ride\n\n"
            + BADGES +
            "Bitcoin's problem has never been the returns — it's the **−80% winters**. The fix "
            "every crypto desk pitches sounds like a thermostat: *\"don't hold a fixed amount of "
            "Bitcoin, hold a fixed amount of **risk**. When BTC gets wild, own less; when it calms "
            "down, own more. Target a constant 30% volatility and you keep the ride while cutting "
            "the heart attacks in half.\"*\n\n"
            "It's a real institutional technique (vol targeting runs trillions in equity land). The "
            "question is whether the port to a single coin that swings between 20% and 150% vol "
            "actually delivers **both** halves of the promise: the smaller crashes *and* the same "
            "ride.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the parameter "
            "grid? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data note.** yfinance daily BTC-USD, 2014 → mid-2026 (11.8 years — one asset, "
            "one history, and BTC is itself the *surviving* coin of its class). Every chart is drawn "
            "by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does it tame the −83% drawdowns? | **Yes — genuinely.** Worst crash goes from "
            f"**{R['b_dd']:.0f}%** to **{R['s_dd']:.0f}%**, and a shuffled-signal test shows that's "
            "*timing skill*, not just holding less Bitcoin (only 2 in 200 random versions do as "
            "well). |\n"
            "| Does it hold vol at 30%? | **Yes.** The strategy's realized vol hugs the target "
            f"(median **{R['vt_median']:.0f}%** vs Bitcoin's own {R['vt_bh_median']:.0f}%). The "
            "thermostat works. |\n"
            "| Is it the *same ride*? | **No.** Over 11.8 years you end with **×"
            f"{R['s_wealth']:.0f}** your money instead of **×{R['b_wealth']:.0f}** — less than a "
            "third of the buy-and-hold wealth. |\n"
            "| Does it *beat* Bitcoin risk-adjusted? | **Can't be certified.** The Sharpe nudges up "
            f"({R['s_sharpe']:.2f} vs {R['b_sharpe']:.2f}) and the alpha is positive "
            f"(+{R['alpha']:.1f}%/yr) but at *t* = {R['t_alpha']:.2f} it never clears the "
            "significance bar. |\n\n"
            "> The honest version of the pitch: **\"half the heart attacks\" is true, \"same "
            "ride\" is false.** You are buying a smoother path, and paying for it with most of the "
            "terminal wealth."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Hold `30% ÷ (BTC's recent volatility)` worth of Bitcoin, rebalanced daily (capped "
            "at 1.5×). When vol runs at 60%, you hold half a position; at 120%, a quarter; in calm "
            "regimes you can lever a little. Same ride, half the heart attacks — the overlay that "
            "tames the −83% drawdowns.\"*\n\n"
            "This is the crypto port of **Moreira & Muir (2017)** — the *volatility-managed "
            "portfolio* result that made vol scaling academically respectable — plus the "
            "institutional folklore of **Harvey et al. (2018)**: vol targeting smooths risk assets "
            "because volatility is *forecastable* while returns are not.\n\n"
            "We test it exactly as pitched: weight = min(1.5, 30% / trailing-30d realized vol), "
            "**yesterday's** vol estimate deciding **today's** position (one-day lag, no peeking), "
            "spare cash earning 0%."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside the pitch, and they deserve different grades:\n\n"
            "1. **Risk control** — *can a mechanical rule hold a wild asset at a chosen volatility "
            "and shrink the crashes?* If yes, that's genuinely useful: it's the difference between "
            "an allocation you can hold and one that shakes you out at the bottom.\n"
            "2. **A free lunch** — *do you keep the same growth while taking half the risk?* That's "
            "an alpha claim. If it were true, vol-targeted BTC would dominate plain BTC, full "
            "stop.\n\n"
            "Sales decks love to blur the two. The tape can separate them."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The tape.** {R['n_days']:,} daily BTC-USD closes, {R['start']} → {R['end']} "
            f"({R['years']:.1f} years — three full crypto winters).\n"
            "- **The rule.** Each day, measure BTC's realized vol over the last 30 days; hold "
            "`min(1.5, 30% / vol)` of the portfolio in BTC, rest in cash at 0%. Yesterday's "
            "measurement sets today's position — one clean lag.\n"
            "- **The race.** Vol-targeted BTC vs plain buy & hold on the *same* days: growth, "
            "volatility, worst drawdown, Sharpe.\n"
            "- **The luck test.** Rebuild the strategy 200 times with the vol signal *shuffled in "
            "time* (same amount of Bitcoin on average, zero information). If shuffled versions "
            "crash just as softly, the \"shield\" was only *holding less*, not timing.\n"
            "- **The knob test.** Re-run everything at 20/30/50% targets and 20/30/60-day windows "
            "— a real effect shouldn't live in one magic parameter."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The ride itself.** Same starting dollar, log scale (each gridline = 10×)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nav_s = (1 + OV['strat']).cumprod(); nav_b = (1 + OV['bh']).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(nav_b.index, nav_b, c=GREY, lw=2, label='buy & hold BTC')\n"
            "    ax.plot(nav_s.index, nav_s, c=GREEN, lw=2, label='vol-targeted 30%')\n"
            "    ax.set_yscale('log')\n"
            "    ax.annotate(f'x{nav_b.iloc[-1]:,.0f}', (nav_b.index[-1], nav_b.iloc[-1]), color=GREY,\n"
            "                xytext=(8,0), textcoords='offset points', va='center')\n"
            "    ax.annotate(f'x{nav_s.iloc[-1]:,.0f}', (nav_s.index[-1], nav_s.iloc[-1]), color=GREEN,\n"
            "                xytext=(8,0), textcoords='offset points', va='center')\n"
            "    ax.set_ylabel('growth of $1 (log scale)')\n"
            "    ax.set_title('Smoother, yes - the same ride, no')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    print(f'terminal wealth: vol-targeted x{nav_s.iloc[-1]:,.1f}  vs  buy & hold x{nav_b.iloc[-1]:,.1f}')\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', 'strat x%.1f vs B&H x%.1f' % (R['s_wealth'], R['b_wealth']))"
        ),
        md(
            f"The green line is visibly calmer — and visibly **lower**: **×{R['s_wealth']:.0f}** vs "
            f"**×{R['b_wealth']:.0f}**. Keep that trade-off in mind while we check whether each "
            "half of the promise holds."
        ),
        md(
            "**Half one: the thermostat.** Does the strategy actually *hold* 30% vol while Bitcoin "
            "swings all over the place?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    roll_s = OV['strat'].rolling(30).std(ddof=1) * np.sqrt(365) * 100\n"
            "    roll_b = OV['bh'].rolling(30).std(ddof=1) * np.sqrt(365) * 100\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(roll_b.index, roll_b, c=GREY, lw=1.4, label='buy & hold BTC')\n"
            "    ax.plot(roll_s.index, roll_s, c=GREEN, lw=1.4, label='vol-targeted strategy')\n"
            "    ax.axhline(30, ls='--', c=RED, lw=1.5, label='30% target')\n"
            "    ax.set_ylabel('rolling 30d realized vol (%/yr)')\n"
            "    ax.set_title('The thermostat works: strategy vol hugs the 30% line')\n"
            "    ax.legend(loc='upper right'); plt.tight_layout(); plt.show()\n"
            "    print(f'strategy vol: median {roll_s.median():.1f}%  |  BTC vol: median {roll_b.median():.1f}%, p90 {roll_b.quantile(.9):.1f}%')\n"
            "else:\n"
            "    print('cache missing - frozen: strategy median %.1f%% vs BTC median %.1f%%' % (R['vt_median'], R['vt_bh_median']))"
        ),
        md(
            f"Confirmed: strategy vol sits at a median **{R['vt_median']:.0f}%** (inside a 20–40% "
            f"band **{R['vt_band']:.0f}%** of days) while raw BTC careens between "
            f"{R['vt_p10']:.0f}% and 100%+. The *mechanical* half of the pitch is simply true.\n\n"
            "**Half two: the heart attacks.** Worst peak-to-trough loss inside each crypto winter."
        ),
        code(
            "names = [c[0] for c in R['crashes']]\n"
            "sdd = [c[1] for c in R['crashes']]; bdd = [c[2] for c in R['crashes']]\n"
            "if HAVE_REAL:\n"
            "    ct = st.crash_table(RET)\n"
            "    names = list(ct.keys()); sdd = [ct[k]['strat'] for k in names]; bdd = [ct[k]['bh'] for k in names]\n"
            "x = np.arange(len(names)); wdt = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(x - wdt/2, bdd, wdt, color=GREY, label='buy & hold')\n"
            "ax.bar(x + wdt/2, sdd, wdt, color=GREEN, label='vol-targeted')\n"
            "for i, v in enumerate(bdd): ax.annotate(f'{v:.0f}%', (x[i]-wdt/2, v), ha='center', va='top', fontsize=9)\n"
            "for i, v in enumerate(sdd): ax.annotate(f'{v:.0f}%', (x[i]+wdt/2, v), ha='center', va='top', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('max drawdown (%)'); ax.set_ylim(-95, 0)\n"
            "ax.set_title('The heart-attack ledger: every winter is roughly cut in half')\n"
            "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "print(dict(zip(names, zip(sdd, bdd))))"
        ),
        md(
            f"Every crash shrinks: **{R['crashes'][0][2]:.0f}% → {R['crashes'][0][1]:.0f}%** in "
            f"2018, **{R['crashes'][2][2]:.0f}% → {R['crashes'][2][1]:.0f}%** in 2021-22, "
            f"**{R['b_dd']:.0f}% → {R['s_dd']:.0f}%** overall. And the quants notebook shows this "
            "is *timing*, not just holding less: 200 shuffled versions of the same rule (same "
            "average Bitcoin, no information) average a **−67%** drawdown — only 2 of 200 match "
            "the real one. \"Half the heart attacks\" earns its badge.\n\n"
            "**Now the price tag.** Same starting dollar — what did each approach turn it into?"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "vals = [R['b_wealth'], R['s_wealth']]\n"
            "if HAVE_REAL:\n"
            "    vals = [float((1+OV['bh']).prod()), float((1+OV['strat']).prod())]\n"
            "ax.bar(['buy & hold', 'vol-targeted 30%'], vals, color=[GREY, GREEN], width=.55)\n"
            "for i, v in enumerate(vals): ax.annotate(f'x{v:,.0f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('growth of $1 over 11.8 years')\n"
            "ax.set_title('The price of the smooth path: less than a third of the wealth')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'wealth kept: {vals[1]/vals[0]*100:.1f}% of buy & hold')"
        ),
        md(
            f"That's the busted half. The overlay kept **{R['kept_cagr']:.0f}%** of the CAGR "
            f"(+{R['s_cagr']:.0f}%/yr vs +{R['b_cagr']:.0f}%/yr — still spectacular), but "
            f"compounding is merciless: 11.8 years of a {R['giveup']:.0f} pp/yr give-up ends at "
            f"**{R['kept_wealth']:.0f}% of the wealth**. \"Same ride\" it is not.\n\n"
            "> 🔬 **For the quants:** the growth gap is *statistically* noise-compatible (HAC "
            f"*t* = {R['t_gap']:.2f} — Bitcoin is that loud), and the Sharpe/alpha edge is positive "
            f"but uncertified (*t* = {R['t_alpha']:.2f}). Neither \"it costs you growth\" nor \"it "
            "adds alpha\" clears the bar; what's certain is the realized arithmetic above."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** *Real on the heart attacks* (drawdowns genuinely halved, "
            "certified as timing by the shuffle test), *weak on the ride* (the risk-adjusted edge "
            f"never clears significance, *t* = {R['t_alpha']:.2f}).\n"
            "- **Tradability — Fragile.** Cheap to run (costs barely dent it) and any exchange can "
            "do it — but what survives scrutiny is **risk control, not extra return**, and it cost "
            f"{R['giveup']:.0f} pp/yr of realized growth on this one 11.8-year tape.\n"
            "- **\"Same ride\"? — Busted.** ×44 vs ×153. Half the heart attacks: nearly literal. "
            "Same ride: no."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Vol targeting is honest about what it is** — a *risk* dial, not a return machine. "
            "If a −50% drawdown is the difference between you holding and you capitulating, giving "
            "up wealth for it can be rational. Just buy it with open eyes.\n"
            "- **The knob is the target.** 20% target → −39% worst crash and a quarter of the "
            "wealth; 50% → −74% and half. There is no setting that keeps the wealth *and* kills "
            "the crash — the grid in the quants notebook makes that trade-off explicit.\n"
            "- **Siblings on this desk.** [210-crypto-trend](../../210-crypto-trend/README.md) tames "
            "the same drawdowns with an in-or-out SMA rule; "
            "[591-vol-managed-portfolio](../../591-vol-managed-portfolio/README.md) tests the same "
            "1/vol recipe on equities, where the alpha claim fares better.\n\n"
            "*Think you can pick the vol target that keeps the ride? The grid says otherwise — "
            "show us a setting with both halves of the promise and we'll re-grade.*"
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
            "# BTC Vol Targeting — a quantitative teardown 🔬\n"
            "### HAC growth-gap + alpha regressions · a 3×3 target/window grid · a 200-seed "
            "shuffled-signal placebo · cost + borrow sweep · a two-world synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — *constant 30% vol target on BTC: same ride, half the heart attacks* — splits "
            "into a **risk claim** (deliver ~30% vol, tame the −83% drawdowns beyond mere exposure "
            "reduction) and a **return claim** (keep the ride / earn vol-timing alpha). We grade "
            "them separately, on the tape.\n\n"
            "> ⚠️ **Data note.** yfinance daily BTC-USD, " + R['start'] + " → " + R['end'] +
            " (4,305 calendar-daily bars, ann = 365; price-only = total-return; rf = 0 on both "
            "legs). Single-asset tape — BTC is itself the surviving coin of its class. Offline core "
            "+ synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R['fingerprint'] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | *Real on the heart attacks:* max DD **{R['s_dd']:.2f}% vs "
            f"{R['b_dd']:.2f}%**, robust in all 9 grid cells, shuffled-signal placebo "
            f"**p = {R['pl_p_dd']:.3f}** (200 seeds). *Weak on the ride:* HAC alpha "
            f"**+{R['alpha']:.2f}%/yr at t = {R['t_alpha']:.2f}** (placebo p = "
            f"{R['pl_p_alpha']:.3f}); grid alpha t = 0.44–1.48, never ≥ 2. |\n"
            f"| **Tradability** | `FRAGILE` | Turnover {R['turnover']:.2f}× NAV/yr; survives 20 bps "
            f"+ 5% borrow (Sharpe {R['costs'][3][2]:.3f}, DD {R['costs'][3][4]:.2f}%) — but the "
            f"certified deliverable is risk control; realized CAGR give-up **{R['giveup']:.2f} "
            "pp/yr**. |\n"
            f"| **Same ride?** | `BUSTED` | Terminal wealth **×{R['s_wealth']:.1f} vs "
            f"×{R['b_wealth']:.1f}** ({R['kept_wealth']:.1f}% kept); growth gap "
            f"{R['gap_ann']:.2f} log-pp/yr (HAC t = {R['t_gap']:.2f}, noise-compatible — the "
            "realized arithmetic is not). |\n\n"
            "> 💡 In plain words: the thermostat and the crash shield are real; the free lunch "
            "is not on this tape."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\hat\\sigma_{t-1}$ be the trailing 30-day realized vol (annualised, "
            "$\\sqrt{365}$) known at the close of day $t-1$. The overlay holds\n\n"
            "$$w_t = \\min\\!\\left(1.5,\\; \\frac{\\sigma^{*}}{\\hat\\sigma_{t-1}}\\right),"
            "\\qquad \\sigma^{*}=30\\%,$$\n\n"
            "earning $r^{strat}_t = w_t r_t - \\text{costs}_t$ with the un-invested fraction at "
            "0%. Exactly one execution lag (`shift(1)`), documented.\n\n"
            "- **H₁ (risk).** The overlay delivers ≈30% realized vol and cuts max drawdown beyond "
            "what *any* same-distribution exposure profile would (timing, not de-risking).\n"
            "- **H₂ (return).** The growth give-up is ≈0 (\"same ride\") and/or the overlay earns "
            "positive risk-adjusted alpha vs B&H (Moreira-Muir logic: vol is forecastable, the "
            "mean is not, so scaling by $1/\\hat\\sigma$ should raise the Sharpe).\n\n"
            "We find **H₁ supported** (vol tracking + DD placebo p = 0.010), **H₂ unsupported** "
            "(alpha t = 1.42; realized give-up 15.5 pp/yr, itself noise-compatible at t = −1.00)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Vol targeting's academic pedigree (Moreira & Muir 2017) was earned on *equity "
            "factors* with a strong leverage effect — vol spikes there carry no extra "
            "compensation, so shedding exposure in high-vol regimes is nearly free. The BTC port "
            "assumes the same disconnect holds for a single asset whose vol regularly triples in "
            "a month. Two inference traps make the naive backtest unreliable:\n\n"
            "1. **Exposure confound.** ANY rule that averages w ≈ 0.6 mechanically shrinks "
            "drawdowns. The shield only counts if it beats *shuffled* versions of itself — same "
            "weight distribution, no alignment with tomorrow's risk (200 seeds; the desk bans "
            "single-seed baselines).\n"
            "2. **Serial correlation.** Daily strategy-minus-B&H differences are autocorrelated "
            "(vol clustering), so every t here is HAC/Newey-West with automatic lag choice.\n\n"
            "The return test races **excess-vs-excess** (rf = 0 both legs, so the convention "
            "cancels), and the wealth question uses the **log-growth gap** "
            "$d_t = \\log(1+r^{strat}_t)-\\log(1+r^{bh}_t)$ — the statistic that compounds to "
            "terminal wealth."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** {R['n_days']:,} daily BTC-USD closes {R['start']} → {R['end']} "
            f"({R['years']:.1f} yrs), as-of pinned, fingerprint `{R['fingerprint']}`; race sample "
            f"{R['n_race']:,} days after the 30-day burn-in.\n"
            "- **Rule.** $w_t = \\min(1.5, 30\\%/\\hat\\sigma_{t-1})$, daily rebalance, one lag; "
            "cash at 0% (conservative against the overlay).\n"
            "- **Headline stats.** CAGR, ann vol, Sharpe (excess-vs-excess), max DD, terminal "
            "wealth; **HAC t on the daily log-growth gap**; **HAC alpha** of strat-on-B&H.\n"
            "- **Robustness.** Full grid: targets {20, 30, 50%} × windows {20, 30, 60d}.\n"
            "- **Placebo.** 200 shuffled-RV seeds → p-values for both the alpha and the DD "
            "shield.\n"
            "- **Costs.** One-way bps × |Δw| × NAV daily (entry excluded, both legs pay it) + "
            "retail borrow spread on max(w−1, 0): 0 / 5 / 10+2% / 20+5%.\n"
            "- **Positive control.** Seeded vol-clustered worlds: risk-priced null (must earn "
            "nothing) vs planted leverage-effect world (must light up), 20 seeds each."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline race and the exposure path\n\n"
            "Weights over time (how much BTC the rule actually holds), then the two legs' full "
            "stats."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "    ax.plot(OV['w'].index, OV['w'], c=GREEN, lw=1.0)\n"
            "    ax.axhline(1.0, ls='--', c=GREY, lw=1.2, label='fully invested (w = 1)')\n"
            "    ax.axhline(1.5, ls=':', c=RED, lw=1.2, label='cap 1.5x')\n"
            "    ax.set_ylabel('weight in BTC'); ax.set_ylim(0, 1.6)\n"
            "    ax.set_title(f'The dial: avg weight {OV[\"avg_w\"]:.3f}, levered {OV[\"share_levered\"]*100:.1f}% of days')\n"
            "    ax.legend(loc='upper right'); plt.tight_layout(); plt.show()\n"
            "    r = st.race(RET)\n"
            "    for leg, p in (('vol-targeted', r['strat']), ('buy & hold ', r['bh'])):\n"
            "        print(f\"{leg}: CAGR {p['cagr_pct']:+7.2f}%  vol {p['vol_ann_pct']:6.2f}%  \"\n"
            "              f\"Sharpe {p['sharpe']:+.3f}  maxDD {p['maxdd_pct']:+7.2f}%  wealth x{p['wealth_mult']:,.1f}\")\n"
            "    print(f\"log-growth gap {r['growth_gap_ann_pct']:+.2f}%/yr  HAC t = {r['t_growth_gap']:+.2f}\")\n"
            "    print(f\"HAC alpha {r['alpha_ann_pct']:+.2f}%/yr  t = {r['t_alpha']:+.2f}  beta {r['beta']:.3f}  appraisal {r['appraisal']:+.3f}\")\n"
            "else:\n"
            "    print('cache missing - frozen:', {k: R[k] for k in ('s_sharpe','b_sharpe','alpha','t_alpha','gap_ann','t_gap')})"
        ),
        md(
            f"> 💡 In plain words: the rule holds **{R['avg_w']:.1f}× ≈ 60%** of a full BTC "
            f"position on average (levered only {R['share_lev']:.0f}% of days), nudges the Sharpe "
            f"from {R['b_sharpe']:.2f} to {R['s_sharpe']:.2f}, and earns a positive but "
            f"**uncertified** alpha (+{R['alpha']:.2f}%/yr, t = {R['t_alpha']:.2f} — below the "
            f"desk's t ≥ 2 bar). Meanwhile the log-growth gap is {R['gap_ann']:.1f} pp/yr against "
            "the overlay."
        ),
        md(
            "### 4b · The placebo — timing or just holding less?\n\n"
            "200 shuffled-RV overlays: identical weight *distribution*, zero alignment with "
            "tomorrow's risk. If the real overlay's drawdown/alpha sit inside the shuffled cloud, "
            "the \"skill\" was mere exposure reduction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_shuffle(RET, n_seeds=200)\n"
            "    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "    a1.hist(pl['alpha_draws'], bins=30, color=GREY, alpha=.85)\n"
            "    a1.axvline(pl['obs_alpha_ann_pct'], c=GREEN, lw=2.5,\n"
            "               label=f\"observed {pl['obs_alpha_ann_pct']:+.1f}%/yr\")\n"
            "    a1.set_xlabel('HAC alpha (%/yr)'); a1.set_ylabel('shuffles')\n"
            "    a1.set_title(f\"Alpha: p = {pl['p_alpha']:.3f} - NOT certified\"); a1.legend()\n"
            "    a2.hist(pl['dd_draws'], bins=30, color=GREY, alpha=.85)\n"
            "    a2.axvline(pl['obs_maxdd_pct'], c=GREEN, lw=2.5,\n"
            "               label=f\"observed {pl['obs_maxdd_pct']:.1f}%\")\n"
            "    a2.set_xlabel('max drawdown (%)')\n"
            "    a2.set_title(f\"DD shield: p = {pl['p_dd']:.3f} - genuine timing\"); a2.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"alpha: obs {pl['obs_alpha_ann_pct']:+.2f}%/yr vs placebo mean {pl['placebo_mean_alpha']:+.2f}%/yr  p={pl['p_alpha']:.3f}\")\n"
            "    print(f\"maxDD: obs {pl['obs_maxdd_pct']:+.2f}% vs placebo mean {pl['placebo_mean_maxdd_pct']:+.2f}%  p={pl['p_dd']:.3f}\")\n"
            "else:\n"
            "    print('cache missing - frozen: p_alpha %.3f  p_dd %.3f' % (R['pl_p_alpha'], R['pl_p_dd']))"
        ),
        md(
            f"> 💡 In plain words: random same-size exposure gets you a **{R['pl_mean_dd']:.0f}%** "
            f"drawdown; the timed rule gets **{R['s_dd']:.1f}%**, and only **2 of 200** shuffles "
            f"match it (p = {R['pl_p_dd']:.3f}) — the shield is *information*, not dilution. The "
            f"alpha, though, is matched by one shuffle in ten (p = {R['pl_p_alpha']:.3f}): not "
            "certifiable. This is the cleanest statement of the split verdict."
        ),
        md(
            "### 4c · The grid — parameter-robust where it's real\n\n"
            "Targets 20/30/50% × windows 20/30/60d. Left: the drawdown cut (real everywhere). "
            "Right: the alpha t (certified nowhere)."
        ),
        code(
            "rows = R['grid']\n"
            "if HAVE_REAL:\n"
            "    g = st.grid(RET)\n"
            "    rows = [(int(x['target']*100), x['window'], x['sharpe'], x['maxdd_pct'],\n"
            "             x['cagr_pct'], x['t_alpha'], x['t_growth_gap']) for x in g]\n"
            "labels = [f\"{r[0]}%/{r[1]}d\" for r in rows]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "a1.bar(labels, [r[3] for r in rows], color=GREEN, width=.6)\n"
            "a1.axhline(R['b_dd'], ls='--', c=RED, lw=1.5, label=f\"B&H {R['b_dd']:.0f}%\")\n"
            "a1.set_ylabel('max drawdown (%)'); a1.set_ylim(-95, 0)\n"
            "a1.tick_params(axis='x', rotation=60)\n"
            "a1.set_title('DD cut: every cell beats -83%'); a1.legend(loc='lower right')\n"
            "a2.bar(labels, [r[5] for r in rows], color=AMBER, width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1.5, label='t = 2 bar')\n"
            "a2.set_ylabel('HAC alpha t'); a2.set_ylim(0, 2.4)\n"
            "a2.tick_params(axis='x', rotation=60)\n"
            "a2.set_title('Alpha t: never certified'); a2.legend(loc='upper right')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in rows:\n"
            "    print(f'target {r[0]:>2}% window {r[1]:>2}d: Sharpe {r[2]:+.3f}  maxDD {r[3]:+7.2f}%  '\n"
            "          f'CAGR {r[4]:+6.2f}%  alpha t {r[5]:+.2f}  gap t {r[6]:+.2f}')"
        ),
        md(
            "> 💡 In plain words: the *risk* result is not a parameter fluke — all nine cells cut "
            "the −83.4% drawdown, smoothly scaled by the target (20% target → −37…−41%, 50% → "
            "−70…−75%). The *return* result is a fluke-shaped nothing — alpha t between 0.44 and "
            "1.48, growth-gap t negative in every cell. There is no knob setting with both halves "
            "of the promise."
        ),
        md(
            "### 4d · Costs & whipsaw — cheap to run, and it doesn't matter\n\n"
            f"Turnover is {R['turnover']:.2f}× NAV/yr; the levered fraction (w > 1 on "
            f"{R['share_lev']:.1f}% of days) pays a retail borrow spread."
        ),
        code(
            "rows = R['costs']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for cb, bs in ((0.0,0.0),(5.0,0.0),(10.0,0.02),(20.0,0.05)):\n"
            "        rc = st.race(RET, cost_bps=cb, borrow_spread_ann=bs)\n"
            "        rows.append((cb, bs*100, rc['strat']['sharpe'], rc['strat']['cagr_pct'],\n"
            "                     rc['strat']['maxdd_pct'], rc['alpha_ann_pct'], rc['t_alpha']))\n"
            "print(f\"{'cost':>6} {'borrow':>7} | {'Sharpe':>7} {'CAGR':>8} {'maxDD':>8} {'alpha':>8} {'t':>6}\")\n"
            "for cb, bs, sh, cg, dd, al, ta in rows:\n"
            "    print(f'{cb:>4.0f}bp {bs:>6.0f}% | {sh:>7.3f} {cg:>+7.2f}% {dd:>+7.2f}% {al:>+7.2f}% {ta:>+6.2f}')"
        ),
        md(
            f"> 💡 In plain words: even at a punitive 20 bps per trade plus 5% margin borrow, the "
            f"Sharpe only drops {R['costs'][0][2]:.3f} → {R['costs'][3][2]:.3f} and the drawdown "
            "moves half a point. Costs are **not** the problem with this strategy; the absent "
            "certified edge is. (What fails the bar gross also fails it net — t goes "
            f"{R['costs'][0][6]:.2f} → {R['costs'][3][6]:.2f}.)"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Two seeded vol-clustered worlds, 20 seeds each: a **risk-priced null** (μ_t ∝ σ_t², "
            "vol timing must earn nothing) and a **planted leverage-effect world** (mean falls as "
            "variance rises — the Moreira-Muir disconnect; the overlay MUST light up)."
        ),
        code(
            "res = [st.synthetic_check(disconnect=dc, n_seeds=20) for dc in (0.0, 2.0)]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "labels = ['NULL\\n(risk priced)', 'PLANTED\\n(leverage effect)']\n"
            "tvals = [r['mean_t'] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(res):\n"
            "    ax.annotate(f\"t={r['mean_t']:.2f}\\n(t>=2: {r['share_t_ge_2']*100:.0f}%)\", (i, max(r['mean_t'], .1)),\n"
            "                ha='center', va='bottom')\n"
            "ax.set_ylabel('mean HAC alpha t (20 seeds)'); ax.set_ylim(0, 5.2)\n"
            "ax.set_title('Control: null stays flat, planted world lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for name, r in zip(('null', 'planted'), res):\n"
            "    print(f\"{name:<8}: mean alpha {r['mean_alpha_ann_pct']:+7.2f}%/yr  mean t {r['mean_t']:+.2f} \"\n"
            "          f\"+/- {r['sd_t']:.2f}  share t>=2 {r['share_t_ge_2']*100:.0f}%\")"
        ),
        md(
            f"> 💡 In plain words: when risk is fairly priced the harness reports nothing (mean "
            f"t = {R['syn_null_t']:+.2f}, {R['syn_null_share']:.0f}% false positives); when a "
            f"genuine vol-return disconnect is planted it fires on every seed (mean "
            f"t = {R['syn_pl_t']:.2f}). The machinery can bank the effect when it exists — so the "
            f"real-tape t = {R['t_alpha']:.2f} is the tape talking, not a blind harness. *(A "
            "faithful-engine / power check only — never cited in support of a stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — *Real on the heart attacks:* max DD **{R['s_dd']:.2f}% vs "
            f"{R['b_dd']:.2f}%**, robust in all nine grid cells, and certified as timing by the "
            f"shuffled-signal placebo (**p = {R['pl_p_dd']:.3f}**, 200 seeds; placebo mean DD "
            f"{R['pl_mean_dd']:.2f}%). *Weak on the ride:* HAC alpha **+{R['alpha']:.2f}%/yr at "
            f"t = {R['t_alpha']:.2f}** (placebo p = {R['pl_p_alpha']:.3f}), grid alpha t "
            "0.44–1.48 — never above the bar. Single-asset survivor tape, named.\n"
            f"- **Tradability `FRAGILE`** — turnover {R['turnover']:.2f}× NAV/yr, unlimited "
            f"capacity, retail-accessible, and robust to 20 bps + 5% borrow (Sharpe "
            f"{R['costs'][3][2]:.3f}). But the certified deliverable is risk control — you pay a "
            f"realized **{R['giveup']:.2f} pp/yr** CAGR give-up for it, and the uncertified Sharpe "
            "edge rests on one 11.8-year tape. Not INVESTABLE as an *edge*; deployable as a "
            "*shield*.\n"
            f"- **Same ride? `BUSTED`** — ×{R['s_wealth']:.1f} vs ×{R['b_wealth']:.1f} terminal "
            f"wealth ({R['kept_wealth']:.1f}% kept); the growth gap ({R['gap_ann']:.2f} "
            f"log-pp/yr) is statistically noise-compatible (HAC t = {R['t_gap']:.2f}) but the "
            "realized arithmetic is decisive: this is a different, smaller ride."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Why equities and not BTC?** Moreira-Muir alpha feeds on a *vol-return "
            "disconnect*. On this BTC tape the disconnect exists but is too noisy to certify in "
            "11.8 years — the point estimate is positive (+7%/yr), the error bars are crypto-"
            "sized. Compare [591-vol-managed-portfolio](../../591-vol-managed-portfolio/README.md), "
            "where 33 years of SPY/QQQ gets the same recipe over the bar (barely).\n"
            "- **Risk overlay vs return engine.** The DD placebo (p = 0.010) is the study's most "
            "durable fact: trailing vol *forecasts* BTC crash risk. That is bankable as position "
            "sizing even if the alpha never certifies.\n"
            "- **The sizing-vs-timing frontier.** [210-crypto-trend](../../210-crypto-trend/"
            "README.md) achieves −70% DD with a binary SMA rule; this overlay gets −53% with "
            "continuous sizing at similar cost. Combining the two signals (trend for direction, "
            "vol for size) is the obvious next study.\n\n"
            "*The reproducible core is offline and deterministic; the signal is trailing 30d "
            "realized vol with one day of lag. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
