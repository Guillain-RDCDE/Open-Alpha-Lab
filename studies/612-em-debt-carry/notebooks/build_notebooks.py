"""Generate the two narrative notebooks for Study 612 (EM Debt Carry).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached dual
(total-return + price-only) tapes under ../_cache/ and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance dual tape,
# EMB window 2008-01-31 -> 2026-06-30, n=222 months; EMLC window 2010-08-31 ->, n=191).
R = dict(
    start="2008-01-31", end="2026-06-30", n_months=222, years=18.5,
    # promised carry
    coupon_emb=4.97, coupon_emb_t=35.2, coupon_ief=2.46,
    pickup_bps=20.24, pickup_ann=2.46, pickup_t=20.09,
    roll_min=1.30, roll_med=2.48, roll_max=3.42,
    # collected carry
    spread_bps=17.83, spread_ann=2.16, spread_t=0.99, skew=-1.72,
    wealth_emb=232.8, wealth_ief=170.2,
    # perf table: (name, cagr, vol, sharpe_x, maxdd)
    perf=[("EMB", 4.67, 11.44, 0.35, -26.74), ("IEF", 2.92, 6.71, 0.27, -23.15),
          ("SPY", 11.24, 15.72, 0.68, -46.32), ("BIL", 1.27, 0.54, None, -0.42)],
    # risk-off profile
    off_bps=-438.0, rest_bps=70.5, gap_bps=-508.5, gap_welch=-4.84, corr_spy=0.69,
    n_off=23, n_rest=199, cut_pct=-5.3,
    # carry premium regression
    alpha_bps=-19.35, alpha_ann=-2.30, alpha_t=-1.44,
    b_ief=0.77, b_ief_t=4.71, b_spy=0.46, b_spy_t=7.85, r2=0.57,
    # EMLC contrast (2010-08 -> 2026-06, n=191)
    emlc_n=191, emlc_start="2010-08-31",
    emlc_coupon=5.60, emlc_pickup=3.29, emlc_pickup_t=21.55,
    emlc_spread_bps=-5.16, emlc_spread_ann=-0.62, emlc_spread_t=-0.24,
    emlc_wealth=118.2, ief_wealth_lc=139.8,
    emb_lc_spread_ann=2.17, emb_lc_spread_t=1.24,
    fx_bill_bps=23.09, fx_bill_ann=2.81, fx_bill_t=1.81,
    # crisis ledger: (label, cum %, months)
    ledger=[("GFC 2008", -17.33, 7), ("Taper tantrum 2013", -2.91, 5),
            ("EM stress 2018", -5.29, 8), ("COVID 2020", -18.79, 3),
            ("Rate/Russia 2022", -9.83, 10)],
    n_crisis=33, n_calm=189, crisis_bps=-164.1, calm_bps=49.6, calm_t=3.38,
    crisis_cum=-54.1, calm_cum=93.7, total_cum=39.6, ledger_welch=-2.15, handback=58,
    # crisis drawdowns: (label, EMB, EMLC or None, IEF, SPY)
    crisis_dd=[("GFC 2008", -34.1, None, -6.0, -46.0),
               ("Taper 2013", -14.5, -15.5, -9.1, -5.6),
               ("EM 2018", -6.4, -16.6, -2.7, -9.9),
               ("COVID 2020", -26.5, -20.6, -4.7, -33.7),
               ("2022", -26.0, -20.1, -17.4, -24.5)],
    # robustness: (label, spread bps, t)
    robust=[("NW lags=3", 17.83, 0.96), ("NW lags=6", 17.83, 0.99),
            ("NW lags=12", 17.83, 1.10), ("ex-GFC 2010+", 16.46, 1.17),
            ("2016-07+", 23.46, 1.23)],
    # synthetic control: (planted spread bps, beta_spy, spread bps, t, gap bps, welch)
    syn=[(0.0, 0.0, 4.54, 0.61, 26.1, 1.21), (30.0, 0.45, 62.91, 3.85, -376.4, -14.41)],
    fingerprint="1bff8a946cdf",
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Survives_the_crises%3F: Mixed](https://img.shields.io/badge/Survives_the_crises%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from em_debt_carry import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    TR, PX = data.load_real()
    MTR, MPX = st.aligned_monthly(TR, PX, ["EMB", "IEF", "SPY", "BIL"])
else:
    TR = PX = MTR = MPX = None
print("real EM-debt cache present:", HAVE_REAL,
      "| months:", (0 if MTR is None else len(MTR)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Emerging-market bonds pay a fat spread. Do you ever get to keep it? 🌍\n"
            "### EM debt carry — a coupon that's absolutely real, and a payoff that mostly isn't, in plain English\n\n"
            + BADGES +
            "Walk into any private bank and you'll hear it: *\"Emerging-market government bonds "
            "yield three to four percent more than US Treasuries. That's carry — income you "
            "collect for taking a bit of exotic risk.\"* And the first half is completely true: "
            "the coupons really are fatter, every single year.\n\n"
            "The question this study asks is the second half: after eighteen and a half years of "
            "defaults, taper tantrums, currency crises, a pandemic and a war — did the people who "
            "held the fund actually **keep** the spread?\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the beta regression and the "
            "crisis ledger? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use the index funds that package the trade — EMB "
            "(dollar-denominated EM bonds, listed just before the 2008 crash) and EMLC (local-"
            "currency EM bonds) — against IEF (US Treasuries of the same ~7-year duration). Index "
            "funds keep their corpses: Venezuela's default, Lebanon's, Russia 2022 are all *in* "
            "these numbers. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the fat coupon real? | **Completely.** EM bonds paid about **+2.5%/yr more "
            "coupon** than Treasuries, in *every* rolling year of the sample. The promise is never "
            "absent. |\n"
            "| Do you get to keep it? | **Not reliably.** The *collected* spread over Treasuries "
            "was +2.2%/yr on average — but so violent in crashes that 18.5 years of data **can't "
            "statistically tell it from zero**. |\n"
            "| When does it hurt? | **Exactly when your stocks hurt.** In the worst 10% of stock-"
            "market months, the EM \"bond\" spread lost **−4.4% per month**. It's equity risk in a "
            "bond wrapper. |\n"
            "| What about local-currency EM bonds? | **Worse.** EMLC paid an even fatter coupon "
            "(+5.6%/yr) and still **lost to Treasuries** over 16 years — the currencies ate all of "
            "it. |\n\n"
            "> The coupon is real. The *keeping* is the problem — and that's the difference "
            "between a yield and a return."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"EM sovereigns yield 300-450 bp over Treasuries. That spread is carry — hold the "
            "bonds, collect the difference, and the occasional default is more than covered.\"*\n\n"
            "It's one of the oldest pitches in fixed income, and it has serious academic backing "
            "on *both* sides: one camp (Klingen-Weder-Zettelmeyer, IMF) found a century of EM "
            "lending earned roughly Treasury returns once defaults were counted; another "
            "(Meyer-Reinhart-Trebesch, *Sovereign Bonds since Waterloo*) found a real 3-4%/yr "
            "premium — punctuated by catastrophes. We drop the debate onto the modern, "
            "actually-investable tape: the ETFs."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Millions of retirement portfolios hold EM debt funds precisely for \"the extra "
            "yield.\" If the spread is genuinely collected, it's a legitimate diversifier that "
            "pays. If it's insurance-selling — steady coupons that get handed back in every "
            "crisis, at the same moment equities fall — then the allocation is just **more of the "
            "risk you already had**, wearing a bond costume. Which one it is changes what the fund "
            "is *for*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"Simple, honest bookkeeping on **{R['years']:.1f} years** of the packaged trade "
            f"({R['start']} → {R['end']}, {R['n_months']} months, 2008 crash included):\n\n"
            "1. **Split the promise from the payoff.** Every fund's return = price move + "
            "distributions. Running the tape twice (total-return and price-only) isolates the "
            "**coupon stream** — the promised carry — from what the price gave back.\n"
            "2. **Match the duration.** EMB's bonds have ~7 years of rate risk; so does IEF. "
            "Comparing them head-to-head cancels the interest-rate bet and leaves the **EM "
            "premium** itself.\n"
            "3. **Condition on the storm.** Look at the spread in the months the stock market "
            "fell hardest, and inside five named crises (2008, 2013, 2018, 2020, 2022): is the "
            "carry still there when everything else is on fire?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the promise vs the payoff.** The coupon pickup EM bonds *promised* over "
            "Treasuries, next to the total-return spread holders actually *collected*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yp = st.yield_pickup(MTR, MPX, 'EMB', 'IEF')\n"
            "    ss = st.spread_stats(MTR, 'EMB', 'IEF')\n"
            "    promised, collected = yp['pickup_ann_pct'], ss['spread_ann_pct']\n"
            "else:\n"
            "    promised, collected = R['pickup_ann'], R['spread_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['PROMISED\\ncoupon pickup','COLLECTED\\ntotal-return spread'], [promised, collected],\n"
            "       color=[GREEN, AMBER], width=.55)\n"
            "for i, v in enumerate([promised, collected]):\n"
            "    ax.annotate(f'{v:+.2f}%/yr', (i, v), ha='center', va='bottom', fontweight='bold')\n"
            "ax.set_ylabel('vs 7-10y Treasuries (%/yr)')\n"
            "ax.set_title('The promise arrives every year; the payoff is another story')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'promised {promised:+.2f}%/yr  vs  collected {collected:+.2f}%/yr')"
        ),
        md(
            f"Looks close — **{R['pickup_ann']:+.2f}%/yr promised, {R['spread_ann']:+.2f}%/yr "
            f"collected** — until you ask how *sure* we are of each number. The coupon pickup is "
            f"so steady it's certain beyond doubt (it was positive in **every** rolling year, "
            f"between +{R['roll_min']:.1f}% and +{R['roll_max']:.1f}%). The collected spread is so "
            f"lumpy that 18.5 years of data **cannot statistically distinguish it from zero** — "
            "the quants notebook puts the certainty score at about **1** where the bar for "
            "\"real\" is **2** (the coupon scores **20**)."
        ),
        md(
            "**Where does the difference go? Watch the wealth race.** $100 in each fund, "
            "total-return, with the five crisis windows shaded."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w_emb = (1 + MTR['EMB']).cumprod() * 100\n"
            "    w_ief = (1 + MTR['IEF']).cumprod() * 100\n"
            "    fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "    ax.plot(w_emb.index, w_emb, color=AMBER, lw=1.8, label='EMB (EM sovereigns, USD)')\n"
            "    ax.plot(w_ief.index, w_ief, color=GREY, lw=1.8, label='IEF (US Treasuries 7-10y)')\n"
            "    for lab, a, b in st.CRISES:\n"
            "        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=RED, alpha=.12)\n"
            "    ax.set_ylabel('$100 grows to (total return)')\n"
            "    ax.set_title('The gap opens in calm years and slams shut in every red band')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final: EMB ${w_emb.iloc[-1]:.1f}  vs  IEF ${w_ief.iloc[-1]:.1f}')\n"
            "else:\n"
            "    print(f\"final: EMB ${R['wealth_emb']:.1f}  vs  IEF ${R['wealth_ief']:.1f} (frozen; cache missing)\")"
        ),
        md(
            f"EMB does finish ahead — **${R['wealth_emb']:.0f} vs ${R['wealth_ief']:.0f}** — but "
            "look at *how*: a staircase up in calm markets, an elevator down in every shaded "
            "crisis. That shape (statisticians call it **skew −1.7**) is why the average can be "
            "positive while the evidence stays inconclusive: a handful of catastrophic months "
            "carries most of the information."
        ),
        md(
            "**And the crashes are not random — they're your stock crashes.** Average monthly "
            "spread in the worst 10% of stock-market months vs all the others."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ro = st.riskoff_profile(MTR, 'EMB', 'IEF')\n"
            "    off, rest = ro['off_bps_mo'] / 100, ro['rest_bps_mo'] / 100\n"
            "else:\n"
            "    off, rest = R['off_bps'] / 100, R['rest_bps'] / 100\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['worst 10% of\\nstock months', 'all other\\nmonths'], [off, rest], color=[RED, GREEN], width=.55)\n"
            "for i, v in enumerate([off, rest]):\n"
            "    ax.annotate(f'{v:+.2f}%/mo', (i, v), ha='center',\n"
            "                va='top' if v < 0 else 'bottom', fontweight='bold')\n"
            "ax.axhline(0, color=GREY, lw=1)\n"
            "ax.set_ylabel('EMB minus IEF (% per month)')\n"
            "ax.set_title('The carry pays in the sun and hands it back in the storm')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'worst-decile stock months: {off:+.2f}%/mo   other months: {rest:+.2f}%/mo')"
        ),
        md(
            f"**{R['off_bps']/100:+.1f}% per month** when stocks are in their worst decile, "
            f"**{R['rest_bps']/100:+.1f}%** the rest of the time. The correlation between the EM "
            f"spread and the stock market is **+{R['corr_spy']:.2f}** — this \"bond diversifier\" "
            "is, to a first approximation, **equity risk with a coupon stapled on**. In fact, once "
            "you account for its Treasury exposure *and* its hidden stock-market exposure, EMB "
            f"delivered slightly **less** than the copycat mix ({R['alpha_ann']:+.1f}%/yr, not "
            "statistically distinguishable from zero) — a do-it-yourself blend of a Treasury fund "
            "and an S&P fund replicated the whole package with nothing left over.\n\n"
            "**One more twist — the local-currency version.** EMLC holds the same governments' "
            f"bonds in *their own* currencies, with an even fatter coupon (**+{R['emlc_coupon']:.1f}%/yr**). "
            f"Over 16 years it turned $100 into **${R['emlc_wealth']:.0f}** while boring old "
            f"Treasuries made **${R['ief_wealth_lc']:.0f}**. The currencies depreciated away the "
            "entire pickup and then some — the *only* version of this carry that ever worked is "
            "the one denominated in dollars."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The coupon pickup is real beyond any doubt "
            f"(**+{R['pickup_ann']:.1f}%/yr**, certainty score ~20). The *collected* spread — the "
            f"actual claim — averaged +{R['spread_ann']:.1f}%/yr but is statistically "
            "indistinguishable from zero after 18.5 years (score ~1, bar = 2).\n"
            "- **Tradability — Mirage.** What you collect is mostly hidden stock-market risk plus "
            "a crisis left tail; a DIY Treasuries+stocks mix matched it with nothing left over, "
            "and the local-currency version lost outright.\n"
            f"- **Survives the crises? — Mixed.** Arithmetically yes (+{R['total_cum']:.0f}% "
            f"cumulative spread remains); statistically no — the five crises clawed back "
            f"**{R['handback']}%** of everything the calm months collected."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **A yield is not a return.** The single transferable lesson: any pitch quoting a "
            "*spread* is quoting the promise, not the payoff. Ask what the *total-return* "
            "difference was, and how it behaved in the worst months.\n"
            "- **The family resemblance.** Fallen-angel bonds "
            "([610](../../610-fallen-angels-premium/README.md)) and mortgage REITs "
            "([611](../../611-mreit-carry/README.md)) tell versions of the same packaged-carry "
            "story; the currency leg that killed EMLC is a whole study of its own "
            "([364](../../364-fx-carry-trade/README.md)).\n"
            "- **Build your own.** Swap IEF for a different duration match, move the crisis "
            "windows, or re-run since 2010 only — the code is beside every chart, and the "
            "conclusion doesn't move.\n\n"
            "*Think the spread is collectable if you just time the exits? The quants notebook "
            "shows the whole premium lives in exactly the months nobody exits in time.*"
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
            "# EM Debt Carry — a quantitative teardown 🔬\n"
            "### Promised-vs-collected decomposition on a dual TR/price-only tape · NW-HAC spread "
            "and equity-beta regression · worst-decile Welch split · a five-window crisis ledger · "
            "the EMLC local-currency contrast · lag/subperiod robustness · synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "EMBI spread is one of the most-quoted numbers in fixed income; the question is "
            "whether the *collected* total-return spread on the investable package clears the "
            "desk's bar — and what it is made of when it doesn't.\n\n"
            "> ⚠️ **Data note.** yfinance daily closes, twice (auto-adjusted TR and price-only), "
            "for EMB / EMLC / IEF / SPY / BIL; headline window 2008-01 → 2026-06 (222 months, GFC "
            "included), EMLC contrast 2010-08 →. Index funds — the defaults are *on* the tape "
            "(survivorship named and dismissed on the Signal axis; the index rulebook's liquidity "
            "screens are the remaining shaping). Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | *Real on the coupon, none on the collected spread*: "
            f"promised pickup **+{R['pickup_ann']:.2f}%/yr** at HAC **t = +{R['pickup_t']:.1f}** "
            f"vs collected TR spread **+{R['spread_ann']:.2f}%/yr** at HAC "
            f"**t = +{R['spread_t']:.2f}** (n = {R['n_months']}, skew {R['skew']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | b_SPY = **+{R['b_spy']:.2f}** (t = +{R['b_spy_t']:.2f}), "
            f"R² = {R['r2']:.2f}; NW-HAC alpha vs IEF+SPY = **{R['alpha_ann']:+.2f}%/yr** "
            f"(t = {R['alpha_t']:+.2f}); worst-decile-SPY spread {R['off_bps']:+.0f} bps/mo "
            f"(Welch t = {R['gap_welch']:+.2f}); EMLC lost to IEF outright "
            f"({R['emlc_spread_ann']:+.2f}%/yr). |\n"
            f"| **Survives the crises?** | `MIXED` | All five windows negative; crisis months "
            f"{R['crisis_bps']:+.0f} bps/mo vs calm {R['calm_bps']:+.1f} (Welch "
            f"t = {R['ledger_welch']:+.2f}); cumulative {R['crisis_cum']:+.1f}% in-crisis vs "
            f"+{R['calm_cum']:.1f}% out; whole-sample t collapses from +{R['calm_t']:.2f} to "
            f"+{R['spread_t']:.2f}. |\n\n"
            "> 💡 In plain words: the coupon is a certainty; keeping it is a coin-flip that "
            "always lands tails in a crisis."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{TR}_{t}$ and $r^{PX}_{t}$ be a fund's monthly total-return and price-only "
            "returns; its **coupon component** is $c_t = r^{TR}_t - r^{PX}_t$. With EMB and the "
            "duration-matched IEF (both ~7y effective duration, both USD):\n\n"
            "$$\\text{promised}_t = c^{EMB}_t - c^{IEF}_t, \\qquad "
            "\\text{collected}_t = r^{TR,EMB}_t - r^{TR,IEF}_t.$$\n\n"
            "- **H₁ (the carry is collected).** $\\overline{\\text{collected}} > 0$ with HAC "
            "*t* ≥ 2 — the desk bar.\n"
            "- **H₂ (it is a premium, not a beta).** The NW-HAC intercept of EMB's excess return "
            "on IEF + SPY excess returns is positive — something is left after duration and "
            "equity risk.\n"
            "- **H₃ (it survives its crises).** The spread's cumulative take inside the named "
            "risk-off windows does not destroy the calm-month harvest.\n\n"
            "We find **H₁ rejected** (t = 0.99), **H₂ rejected with the sign flipped** (alpha "
            "−2.30%/yr, t = −1.44, b_SPY = +0.46 at t = +7.85), **H₃ split** (arithmetic yes, "
            "+39.6% cumulative; statistical no, the five windows claw back 58% and the t with it)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The EMBI spread (300-450 bp historically) is the most-quoted carry number in EM "
            "investing, and the literature genuinely splits: Klingen-Weder-Zettelmeyer (2004) "
            "find a century of EM lending returned roughly Treasuries after defaults; "
            "Meyer-Reinhart-Trebesch (2022) find +3-4%/yr *including* defaults. Both can be "
            "right about their samples — the question for an allocator is what the **investable, "
            "post-2007 package** did. Borri-Verdelhan (2011) supply the null we test against: EM "
            "sovereign returns are compensation for *systematic* (equity-like) risk, taken at "
            "the worst times. If they're right, the spread should (a) fail unconditional "
            "significance despite a positive mean, (b) load on SPY with no residual alpha, and "
            "(c) concentrate its losses in equity crash months. All three predictions hit."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** Dual yfinance closes (TR + price-only), EMB/EMLC/IEF/SPY/BIL; headline "
            f"window {R['start']} → {R['end']} ({R['n_months']} months); as-of frozen, partial "
            "month dropped; fingerprint `" + R["fingerprint"] + "`.\n"
            "- **Duration match.** EMB ≈ 7y effective duration vs IEF 7-8y — the TR spread "
            "isolates credit/carry, not a rates bet.\n"
            "- **Inference.** NW-HAC *t* (6 lags; 3/12 as robustness) on every monthly mean — "
            "bond spreads are serially correlated. Welch *t* on the risk-off group split. "
            "Excess-vs-excess Sharpe (over BIL).\n"
            "- **Execution.** Passive buy-and-hold decomposition; entry at the first month-end "
            "close of the common window (one documented, trivial lag — no timing signal exists "
            "to lag). Expense ratios (0.39%/0.30%) are inside the tape; two one-way trades over "
            "18.5y ≈ 0.5 bps/yr amortised.\n"
            "- **Crisis windows.** Five fixed calendar windows (2008, 2013, 2018, 2020, 2022) "
            "documented from the events, not fitted to the series.\n"
            "- **Positive control.** A deterministic synthetic world with planted spread / "
            "coupon / risk-off-beta knobs: the null must stay silent, the plants must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Promised vs collected — two t-stats, one tape\n\n"
            "The coupon pickup (TR-minus-price-only, EMB minus IEF) against the collected TR "
            "spread, each with its HAC *t*. Same 222 months, same funds — and a factor-20 gap in "
            "evidence."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yp = st.yield_pickup(MTR, MPX, 'EMB', 'IEF')\n"
            "    ss = st.spread_stats(MTR, 'EMB', 'IEF')\n"
            "    vals = [(yp['pickup_ann_pct'], yp['hac_t']), (ss['spread_ann_pct'], ss['hac_t'])]\n"
            "else:\n"
            "    vals = [(R['pickup_ann'], R['pickup_t']), (R['spread_ann'], R['spread_t'])]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "labs = ['promised\\n(coupon pickup)', 'collected\\n(TR spread)']\n"
            "a1.bar(labs, [v[0] for v in vals], color=[GREEN, AMBER], width=.55)\n"
            "for i, v in enumerate(vals): a1.annotate(f'{v[0]:+.2f}%/yr', (i, v[0]), ha='center', va='bottom')\n"
            "a1.set_ylabel('%/yr vs IEF'); a1.set_title('Magnitudes: nearly the same')\n"
            "a2.bar(labs, [v[1] for v in vals], color=[GREEN, AMBER], width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, v in enumerate(vals): a2.annotate(f't = {v[1]:+.2f}', (i, v[1]), ha='center', va='bottom')\n"
            "a2.set_ylabel('HAC t (NW, 6 lags)'); a2.set_title('Evidence: a factor of 20 apart'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'promised {vals[0][0]:+.2f}%/yr (t {vals[0][1]:+.2f})   collected {vals[1][0]:+.2f}%/yr (t {vals[1][1]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the *promise* (+{R['pickup_ann']:.2f}%/yr, t = "
            f"+{R['pickup_t']:.1f}, positive in every rolling year between +{R['roll_min']:.1f}% "
            f"and +{R['roll_max']:.1f}%) is one of the most certain numbers on the desk. The "
            f"*payoff* (+{R['spread_ann']:.2f}%/yr, t = +{R['spread_t']:.2f}, skew "
            f"{R['skew']:.2f}) doesn't clear half the bar. A yield is not a return.\n"
        ),
        md(
            "### 4b · What the collected spread is made of — the beta regression\n\n"
            "NW-HAC regression of EMB's excess return (over BIL) on IEF and SPY excess returns. "
            "If the carry is a genuine premium, the intercept is positive; if it is packaging, "
            "the betas absorb everything."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.carry_premium(MTR, 'EMB')\n"
            "else:\n"
            "    cp = dict(alpha_bps_mo=R['alpha_bps'], alpha_ann_pct=R['alpha_ann'], t_alpha=R['alpha_t'],\n"
            "              beta_IEF=R['b_ief'], t_IEF=R['b_ief_t'], beta_SPY=R['b_spy'], t_SPY=R['b_spy_t'], r2=R['r2'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "names = ['alpha\\n(bps/mo)', 'b_IEF x10', 'b_SPY x10']\n"
            "vals = [cp['alpha_bps_mo'], cp['beta_IEF'] * 10, cp['beta_SPY'] * 10]\n"
            "ts = [cp['t_alpha'], cp['t_IEF'], cp['t_SPY']]\n"
            "cols = [RED if v < 0 else GREEN for v in vals]\n"
            "ax.bar(names, vals, color=cols, width=.5)\n"
            "for i, (v, t) in enumerate(zip(vals, ts)):\n"
            "    ax.annotate(f'{v:+.2f}\\n(t {t:+.2f})', (i, v), ha='center', va='bottom' if v >= 0 else 'top')\n"
            "ax.axhline(0, color=GREY, lw=1)\n"
            "ax.set_title(f\"EMB excess = alpha + b.IEF + b.SPY  (R2 = {cp['r2']:.2f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"alpha {cp['alpha_bps_mo']:+.2f} bps/mo ({cp['alpha_ann_pct']:+.2f}%/yr) t={cp['t_alpha']:+.2f} | \"\n"
            "      f\"b_IEF {cp['beta_IEF']:+.2f} (t {cp['t_IEF']:+.2f}) b_SPY {cp['beta_SPY']:+.2f} (t {cp['t_SPY']:+.2f})\")"
        ),
        md(
            f"> 💡 In plain words: EMB ≈ **0.77 × Treasuries + 0.46 × stock market − "
            f"{abs(R['alpha_ann']):.1f}%/yr**. More than half the variance (R² = {R['r2']:.2f}) "
            f"is duration plus equity beta; the stock loading is unambiguous (t = "
            f"+{R['b_spy_t']:.2f}); and the residual \"carry premium\" is *negative* though not "
            "significantly so. A DIY IEF+SPY blend replicated the package with nothing left "
            "over — that is the Mirage stamp, in one regression.\n"
        ),
        md(
            "### 4c · The carry-crash profile — the Welch split and the skew\n\n"
            "Monthly spread conditional on SPY's worst decile vs the rest, plus the histogram "
            "that shows where the information lives."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ro = st.riskoff_profile(MTR, 'EMB', 'IEF')\n"
            "    sp = (MTR['EMB'] - MTR['IEF']) * 100\n"
            "else:\n"
            "    ro = dict(off_bps_mo=R['off_bps'], rest_bps_mo=R['rest_bps'], gap_bps_mo=R['gap_bps'],\n"
            "              welch_t=R['gap_welch'], corr_spy=R['corr_spy'], n_off=R['n_off'], n_rest=R['n_rest'])\n"
            "    rng = np.random.default_rng(612); sp = pd.Series(rng.normal(0.18, 2.2, 222))  # illustrative only\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(['worst 10%\\nSPY months', 'all other\\nmonths'],\n"
            "       [ro['off_bps_mo'] / 100, ro['rest_bps_mo'] / 100], color=[RED, GREEN], width=.55)\n"
            "for i, v in enumerate([ro['off_bps_mo'] / 100, ro['rest_bps_mo'] / 100]):\n"
            "    a1.annotate(f'{v:+.2f}%/mo', (i, v), ha='center', va='top' if v < 0 else 'bottom')\n"
            "a1.axhline(0, color=GREY, lw=1); a1.set_ylabel('EMB - IEF (%/mo)')\n"
            "a1.set_title(f\"Conditional split: Welch t = {ro['welch_t']:+.2f}\")\n"
            "a2.hist(sp.dropna(), bins=40, color=AMBER, alpha=.9)\n"
            "a2.axvline(0, color=GREY, lw=1)\n"
            "a2.set_xlabel('monthly TR spread (%)'); a2.set_ylabel('months')\n"
            "a2.set_title('The left tail is the whole story (skew ' + f\"{R['skew']:.2f}\" + ')')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"off {ro['off_bps_mo']:+.1f} bps/mo vs rest {ro['rest_bps_mo']:+.1f} | gap {ro['gap_bps_mo']:+.1f} \"\n"
            "      f\"(Welch t {ro['welch_t']:+.2f}) | corr(spread, SPY) {ro['corr_spy']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: {R['n_off']} months — the stock market's worst decile — cost "
            f"the spread **{R['off_bps']/100:+.1f}% per month** while the other {R['n_rest']} "
            f"paid +{R['rest_bps']/100:.1f}%. The gap is {R['gap_bps']:+.0f} bps/mo at Welch "
            f"t = {R['gap_welch']:+.2f}, and the spread-SPY correlation is +{R['corr_spy']:.2f}. "
            "This is the Brunnermeier-Nagel-Pedersen carry-crash signature on a bond fund: "
            "up the stairs, down the elevator, and the elevator is wired to your equity "
            "allocation.\n"
        ),
        md(
            "### 4d · The third axis — the five-crisis ledger\n\n"
            "Cumulative monthly spread inside each fixed risk-off window, and the "
            "crisis-vs-calm ledger that decides whether the spread survives the events it "
            "insures."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cl = st.crisis_ledger(MTR, 'EMB', 'IEF')\n"
            "    per = cl['per_window']\n"
            "else:\n"
            "    cl = dict(crisis_cum_pct=R['crisis_cum'], calm_cum_pct=R['calm_cum'], total_cum_pct=R['total_cum'],\n"
            "              crisis_bps_mo=R['crisis_bps'], calm_bps_mo=R['calm_bps'], welch_t=R['ledger_welch'],\n"
            "              calm_hac_t=R['calm_t'], n_crisis=R['n_crisis'], n_calm=R['n_calm'])\n"
            "    per = R['ledger']\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "labs = [p[0] for p in per] + ['CALM months\\n(all 189)']\n"
            "vals = [p[1] for p in per] + [cl['calm_cum_pct']]\n"
            "cols = [RED] * len(per) + [GREEN]\n"
            "ax.bar(labs, vals, color=cols, width=.6)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, color=GREY, lw=1)\n"
            "ax.set_ylabel('cumulative TR spread (%)')\n"
            "ax.set_title('Every named crisis is negative; the calm months pay for all of them')\n"
            "plt.xticks(rotation=15); plt.tight_layout(); plt.show()\n"
            "print(f\"crisis cum {cl['crisis_cum_pct']:+.1f}% ({cl['n_crisis']} mo, {cl['crisis_bps_mo']:+.1f} bps/mo)  \"\n"
            "      f\"calm cum {cl['calm_cum_pct']:+.1f}% ({cl['n_calm']} mo, {cl['calm_bps_mo']:+.1f} bps/mo, HAC t {cl['calm_hac_t']:+.2f})\")\n"
            "print(f\"total {cl['total_cum_pct']:+.1f}%  | crisis-vs-calm Welch t = {cl['welch_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: all five windows are negative — {R['n_crisis']} crisis months "
            f"surrender **{R['crisis_cum']:+.1f}%** of the **+{R['calm_cum']:.1f}%** the "
            f"{R['n_calm']} calm months collected ({R['handback']}% handed back), leaving "
            f"**+{R['total_cum']:.1f}%** overall. Calm-only the spread is decisively real (HAC "
            f"t = +{R['calm_t']:.2f}); all-in it is t = +{R['spread_t']:.2f}. The spread survives "
            "**arithmetically, not statistically** — hence the grey MIXED, spelled out.\n"
        ),
        md(
            "### 4e · The local-currency contrast + robustness\n\n"
            "EMLC (same sovereigns, local currency, fatter coupon) vs IEF on the common 2010-08+ "
            "window; then the HAC-lag and subperiod sweep on the headline spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    MTR2, MPX2 = st.aligned_monthly(TR, PX, ['EMLC', 'EMB', 'IEF', 'SPY', 'BIL'])\n"
            "    s_lc = st.spread_stats(MTR2, 'EMLC', 'IEF'); s_hc = st.spread_stats(MTR2, 'EMB', 'IEF')\n"
            "    y_lc = st.yield_pickup(MTR2, MPX2, 'EMLC', 'IEF')\n"
            "    rows = [('EMLC - IEF', s_lc['spread_ann_pct'], s_lc['hac_t']),\n"
            "            ('EMB - IEF', s_hc['spread_ann_pct'], s_hc['hac_t'])]\n"
            "    print(f\"EMLC coupon {y_lc['coupon_name_ann_pct']:+.2f}%/yr (pickup t {y_lc['hac_t']:+.1f}) — \"\n"
            "          f\"but TR spread {s_lc['spread_ann_pct']:+.2f}%/yr (t {s_lc['hac_t']:+.2f})\")\n"
            "else:\n"
            "    rows = [('EMLC - IEF', R['emlc_spread_ann'], R['emlc_spread_t']),\n"
            "            ('EMB - IEF', R['emb_lc_spread_ann'], R['emb_lc_spread_t'])]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar([r[0] for r in rows], [r[1] for r in rows],\n"
            "       color=[RED if r[1] < 0 else AMBER for r in rows], width=.5)\n"
            "for i, r in enumerate(rows):\n"
            "    ax.annotate(f'{r[1]:+.2f}%/yr\\n(t {r[2]:+.2f})', (i, r[1]), ha='center',\n"
            "                va='top' if r[1] < 0 else 'bottom')\n"
            "ax.axhline(0, color=GREY, lw=1)\n"
            "ax.set_ylabel('TR spread vs IEF (%/yr), 2010-08+')\n"
            "ax.set_title('Local currency: fatter coupon, negative payoff')\n"
            "plt.tight_layout(); plt.show()\n"
            "if HAVE_REAL:\n"
            "    spq = MTR['EMB'] - MTR['IEF']\n"
            "    for lags in (3, 6, 12):\n"
            "        mu, t = st.nw_tstat(spq, lags=lags)\n"
            "        print(f'NW lags={lags:>2}: {mu*1e4:+.2f} bps/mo  t = {t:+.2f}')\n"
            "    for lab, lo in [('ex-GFC 2010+', '2010-01-01'), ('2016-07+', '2016-07-01')]:\n"
            "        mu, t = st.nw_tstat(spq.loc[lo:])\n"
            "        print(f'{lab:<14}: {mu*1e4:+.2f} bps/mo  t = {t:+.2f}')\n"
            "else:\n"
            "    for lab, mu, t in R['robust']: print(f'{lab:<14}: {mu:+.2f} bps/mo  t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: EMLC's coupon is even fatter (+{R['emlc_coupon']:.1f}%/yr, "
            f"pickup t ≈ 22) and its payoff is **negative** ({R['emlc_spread_ann']:+.2f}%/yr; "
            f"$100 → ${R['emlc_wealth']:.0f} vs ${R['ief_wealth_lc']:.0f} for IEF) — the FX leg "
            f"bills ~{R['fx_bill_ann']:.1f}%/yr (EMB−EMLC, t = +{R['fx_bill_t']:.2f}); see "
            "[364-fx-carry-trade](../../364-fx-carry-trade/README.md) for why. And no HAC lag "
            f"(t = 0.96-1.10) or subperiod (ex-GFC t = 1.17, last decade t = 1.23) rescues the "
            "hard-currency spread either. Du-Schreger (2016) is the formal account of the "
            "local-currency wedge.\n"
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic synthetic world with planted knobs: `spread` (the TR spread), "
            "`coupon` (the pickup, planted at 25 bps/mo) and `beta_spy` (the crash coupling). "
            "The null must stay silent; the plants must light up."
        ),
        code(
            "res = []\n"
            "for spread, beta_spy in [(0.0, 0.0), (0.0030, 0.45)]:\n"
            "    ttr, tpx = data.synthetic_world(spread=spread, beta_spy=beta_spy, seed=612)\n"
            "    smtr, smpx = st.aligned_monthly(ttr, tpx, ['EMD', 'IEF', 'SPY', 'BIL'])\n"
            "    sss = st.spread_stats(smtr, 'EMD', 'IEF')\n"
            "    syp = st.yield_pickup(smtr, smpx, 'EMD', 'IEF')\n"
            "    sro = st.riskoff_profile(smtr, 'EMD', 'IEF')\n"
            "    res.append((spread * 1e4, beta_spy, sss['spread_bps_mo'], sss['hac_t'],\n"
            "                syp['pickup_bps_mo'], syp['hac_t'], sro['gap_bps_mo'], sro['welch_t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labs = [f'planted spread {r[0]:.0f} bps\\nbeta_spy {r[1]:.2f}' for r in res]\n"
            "ax.bar(labs, [r[3] for r in res], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(res): ax.annotate(f't = {r[3]:+.2f}', (i, r[3]), ha='center', va='bottom')\n"
            "ax.set_ylabel('spread HAC t')\n"
            "ax.set_title('Control: null stays silent, planted spread lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for r in res:\n"
            "    print(f'planted {r[0]:+5.1f} bps/mo, b_spy {r[1]:.2f}: spread {r[2]:+.2f} (t {r[3]:+.2f}) | '\n"
            "          f'pickup {r[4]:+.2f} (t {r[5]:+.1f}, planted 25) | risk-off gap {r[6]:+.1f} (Welch {r[7]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with nothing planted the spread test reads t = "
            f"+{R['syn'][0][3]:.2f} and the risk-off split t = +{R['syn'][0][5]:.2f} — silence; "
            f"a planted 30 bps/mo spread reads t = +{R['syn'][1][3]:.2f} and a planted 0.45 "
            f"equity beta produces the crash gap at Welch t = {R['syn'][1][5]:.2f}. The machinery "
            "detects what exists and invents nothing — so the real-tape t = 0.99 is a genuine "
            "absence, not low power alone (the same 222-month design banks a planted 30 bps/mo). "
            "*(A faithful-engine / power check only — never cited to support the real-tape "
            "stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — *Real on the coupon, none on the collected spread.* Promised "
            f"pickup **+{R['pickup_ann']:.2f}%/yr** at HAC **t = +{R['pickup_t']:.1f}** (positive "
            f"in every rolling year); collected TR spread **+{R['spread_ann']:.2f}%/yr** at HAC "
            f"**t = +{R['spread_t']:.2f}** (skew {R['skew']:.2f}), unrescued by any lag or "
            "subperiod. Index funds keep their defaults, so survivorship is not the out.\n"
            f"- **Tradability `MIRAGE`** — the package is duration plus hidden equity beta "
            f"(b_SPY = +{R['b_spy']:.2f} at t = +{R['b_spy_t']:.2f}, R² = {R['r2']:.2f}) with a "
            f"*negative* residual alpha ({R['alpha_ann']:+.2f}%/yr, t = {R['alpha_t']:+.2f}) and "
            f"a crash profile wired to equities ({R['off_bps']:+.0f} bps/mo in the worst SPY "
            f"decile, Welch t = {R['gap_welch']:+.2f}). EMLC lost to Treasuries outright. Access "
            "and costs are fine; the *content* is the mirage.\n"
            f"- **Survives the crises? `MIXED`** — arithmetically yes (+{R['total_cum']:.1f}% "
            f"cumulative survives all five windows); statistically no (every window negative, "
            f"{R['handback']}% of calm-month carry handed back, whole-sample t collapses from "
            f"+{R['calm_t']:.2f} to +{R['spread_t']:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The packaged-carry family.** [610-fallen-angels-premium]"
            "(../../610-fallen-angels-premium/README.md) (credit selection inside high yield) and "
            "[611-mreit-carry](../../611-mreit-carry/README.md) (levered MBS net-interest-margin) "
            "run the same dual-tape decomposition on different wrappers; this study is the "
            "sovereign-credit member, and the cleanest case of *coupon real, payoff beta*.\n"
            "- **The FX leg deserves its own trial.** The EMB−EMLC wedge (~2.8%/yr) is the "
            "currency carry problem in miniature — [364-fx-carry-trade]"
            "(../../364-fx-carry-trade/README.md) prosecutes it directly.\n"
            "- **Where the debate stays open.** Meyer-Reinhart-Trebesch's two centuries suggest "
            "the premium pays over horizons that include *recoveries from* default, not just the "
            "defaults; our 18.5-year window contains five crises and only partial recoveries. A "
            "longer tape could still redeem H₁ — the desk grades the tape it has.\n\n"
            "*The reproducible core is offline and deterministic; the signal is the EMB−IEF "
            "total-return spread against the coupon pickup, with the five-crisis ledger as the "
            "myth-check. Methods and sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
