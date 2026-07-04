"""Generate the two narrative notebooks for Study 617 (Crash-Insurance-Cost).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
TAIL/IEF/SPY/^VIX tape under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic control runs anywhere, no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance TAIL/IEF/SPY/^VIX,
# TAIL era 2017-04-06 -> 2026-06-30, as-of 2026-06-30, fingerprint c3658cfa4e95).
R = dict(
    asof="2026-06-30", fingerprint="c3658cfa4e95",
    start="2017-04-06", end="2026-06-30", years=9.2, n_days=2319, n_months=110,
    total_pct=-49.7, cagr_pct=-7.17, vol_pct=14.8,
    daily_bps=-2.53, daily_t=-1.49, monthly_bps=-56.52, monthly_t=-2.17,
    alpha_d_bps=-3.08, alpha_d_ann=-7.75, alpha_d_t=-1.98, beta_d=1.05, r2_d=0.226,
    alpha_m_bps=-64.09, alpha_m_ann=-7.69, alpha_m_t=-3.28, beta_m=0.73,
    pre_ann=-14.70, pre_t=-2.50, post_ann=-8.07, post_t=-1.75,
    covid_gain=28.47, covid_peak="2020-04-02", spy_fall=-33.72,
    giveback="2021-03-05", giveback_days=337, holder_peak=4.22, holder_now=-49.67,
    vrp_start="1993-02", vrp_end="2026-06", vrp_months=401,
    vrp_var=-0.95, vrp_var_t=-3.12, vrp_vol=-3.79, vrp_vol_t=-9.42, vrp_win=16.2,
    iv=20.98, rv=18.58,
    vrp_era_var=-0.67, vrp_era_var_t=-1.06, vrp_era_vol=-3.62, vrp_era_vol_t=-4.44,
    n_cohorts=110, ahead_now=1, ahead_share=0.9, best=0.10, best_date="2025-01-31",
    median=-34.1, worst=-49.4, ever_ahead=71.8,
    # blends: (w_tail %, blend CAGR %, SPY CAGR %, drag pp/yr, blend MDD, SPY MDD,
    #          blend covid DD, SPY covid DD) — 5 bps one-way x turnover
    blends=[(5, 14.06, 15.12, 1.06, -22.9, -23.9, -17.6, -19.4),
            (10, 12.99, 15.12, 2.13, -21.9, -23.9, -15.8, -19.4),
            (20, 10.84, 15.12, 4.28, -19.9, -23.9, -12.1, -19.4)],
    # synthetic control: (planted bleed %/yr, mean alpha %/yr, mean HAC t, % flagged)
    ctrl=[(0.0, -0.21, -0.31, 5.0), (5.0, -5.21, -5.81, 100.0)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![2020 win kept?: Busted](https://img.shields.io/badge/2020_win_kept%3F-Busted-8b949e?style=flat-square)\n\n"
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

from crash_insurance_cost import data, strategy as st

HAVE_REAL = data.have_real()
TAPE = data.load_real() if HAVE_REAL else None
print("real cache present:", HAVE_REAL, "| rows:", 0 if TAPE is None else len(TAPE))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# What does crash insurance actually cost? ☂️\n"
            "### TAIL, the fund that buys the puts everyone else sells — in plain English\n\n"
            + BADGES +
            "There is an ETF whose whole job is to protect you from a market crash. It's called "
            "**TAIL**: roughly 90% safe US Treasury bonds plus a steady budget of deep out-of-the-money "
            "**put options** on the S&P 500 — lottery tickets that pay off when the market falls hard. "
            "It launched in April 2017, it did **exactly what its prospectus promised** — and a dollar "
            "invested at launch is worth about **50 cents** today.\n\n"
            "That's not a scandal. That's the **price of insurance**, quoted in public, every day. This "
            "study measures the bill precisely — and shows the same money landing in the pocket of the "
            "desk's short-volatility studies ([92-easy-money](../../92-easy-money/), "
            "[63-free-fall](../../63-free-fall/)), which sit on the *other side* of this exact trade.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the regression and the variance "
            "arithmetic? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it from cached real data (as-of "
            + R["asof"] + "); house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does crash insurance really bleed every year? | **Yes — measurably.** TAIL has lost "
            f"**{R['cagr_pct']:.1f}% a year** since 2017 ({R['total_pct']:.0f}% in total), and about "
            f"**{abs(R['alpha_m_ann']):.0f}% a year** of that is the *insurance itself* (the puts + the "
            "fee), not its bonds. Statistically solid, not bad luck. |\n"
            f"| But didn't it pay off in 2020? | **Spectacularly — for 337 days.** It jumped "
            f"**+{R['covid_gain']:.0f}%** while the market fell a third… and by March 2021 the entire "
            "jackpot was gone. |\n"
            f"| Did *anyone* who bought it come out ahead? | **Essentially no.** Of {R['n_cohorts']} "
            f"possible monthly entry points since launch, **{R['ahead_now']}** is above water today (by "
            f"+{R['best']:.1f}%, bought in {R['best_date'][:4]} — *after* the crash). |\n"
            "| Where does the money go? | To the **sellers** of the insurance. Option markets "
            f"consistently price volatility ~{R['iv'] - R['rv']:.1f} points higher than what the market "
            "then delivers — that gap is the *volatility risk premium*, and the buyer pays it in 5 "
            "months out of 6. |\n\n"
            "> Crash insurance isn't broken — it's **expensive, and priced to be**. The claim is true, "
            "and the money is real: it just flows the other way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Tail-risk funds and put-buying strategies bleed money every single year. The bleed "
            "isn't a flaw — it IS the volatility risk premium, seen from the buyer's side.\"*\n\n"
            "Insurance intuition says this should be true: home insurers profit, so homeowners — in "
            "aggregate — pay more than they collect. The market version: people *really* dislike "
            "crashes, so crash protection trades **above** its actuarial value, permanently. Academics "
            "have measured it for decades (options that protect are the most reliably 'overpriced' "
            "instruments in finance). TAIL is the rare **live, buyable, fee-charging** version — the "
            "cleanest public quote of what the insurance costs."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Millions of investors hold a slice of \"tail protection\" because the 2020-spike chart is "
            "the best marketing image in finance. If the bleed is real and large, that slice quietly "
            "costs several percent **per year** — far more than the crash relief is worth to most "
            "portfolios. And if the bleed *is* the volatility risk premium, then the desk's short-vol "
            "studies (92, 63 — both graded **Real**) and this fund are the two ends of one pipe: "
            "whatever TAIL's holders lose, volatility sellers collect."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"Four measurements on real prices ({R['start']} → {R['end']}, dividends reinvested):\n\n"
            "1. **The raw bill.** TAIL's total-return path since launch, with proper "
            "serial-correlation-robust statistics on the drift.\n"
            "2. **The bill, itemised.** TAIL is ~90% Treasury bonds + puts. Compare it against **IEF** "
            "(the same bonds, without the puts): the gap is what the *insurance itself* costs — bond "
            "moves can't be blamed.\n"
            "3. **The one payoff.** Quantify 2020 end-to-end: how big, how long it lasted, who kept it.\n"
            "4. **Name the premium.** From the VIX (the market's quoted price of insurance) vs what the "
            "S&P actually delivered, month by month since 1993 — the insurance buyer's win/loss record."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The whole story in one chart.** TAIL since launch, with the 2020 spike — and what "
            "happened after."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = TAPE['tail'].dropna(); curve = px / px.iloc[0]\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(curve.index, curve.values, color=RED, lw=1.6, label='TAIL (total return, net of fee)')\n"
            "    ax.axhline(1.0, color=GREY, ls=':', lw=1)\n"
            "    peak = pd.Timestamp(R['covid_peak']); gb = pd.Timestamp(R['giveback'])\n"
            "    ax.annotate(f\"COVID spike: +{R['covid_gain']:.0f}% in 6 weeks\\n(holder since 2017: just +{R['holder_peak']:.1f}%)\",\n"
            "                xy=(peak, float(curve.loc[:peak].iloc[-1])), xytext=(peak, 1.35),\n"
            "                arrowprops=dict(arrowstyle='->', color=GREY), fontsize=9)\n"
            "    ax.annotate(f\"jackpot fully given back\\n({R['giveback_days']} days later)\",\n"
            "                xy=(gb, float(curve.loc[:gb].iloc[-1])), xytext=(gb, 1.22),\n"
            "                arrowprops=dict(arrowstyle='->', color=GREY), fontsize=9)\n"
            "    ax.set_ylabel('growth of $1 at launch'); ax.legend(loc='lower left')\n"
            "    ax.set_title(f\"Crash insurance, lived: $1 in 2017 is ${float(curve.iloc[-1]):.2f} today ({R['cagr_pct']:.1f}%/yr)\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'total {R[\"total_pct\"]:.1f}%  CAGR {R[\"cagr_pct\"]:.2f}%/yr  over {R[\"years\"]:.1f} years')\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', R['total_pct'], '% total,', R['cagr_pct'], '%/yr')"
        ),
        md(
            f"A dollar at launch is **50 cents** today: **{R['cagr_pct']:.1f}%/yr** for "
            f"{R['years']:.0f} straight years, *including* the best crash payoff the fund will likely "
            "ever see. And the fund did nothing wrong — this is the product working as designed.\n\n"
            "**Itemise the bill.** TAIL is mostly Treasury bonds. Strip the bond part out (compare "
            "against IEF, the same kind of bonds) and what's left is the pure cost of the puts + fee:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = st.to_returns(TAPE)\n"
            "    dm = st.decompose_monthly(TAPE)\n"
            "    # counterfactual: TAIL's bond sleeve alone (beta x IEF), vs actual TAIL\n"
            "    m = (1.0 + rets).resample('ME').prod() - 1.0\n"
            "    m = m.iloc[1:]\n"
            "    bond_part = (1.0 + dm['beta_ief'] * m['ief']).cumprod()\n"
            "    actual = (1.0 + m['tail']).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(bond_part.index, bond_part.values, color=GREY, lw=1.6, label='its bond sleeve alone (beta x IEF)')\n"
            "    ax.plot(actual.index, actual.values, color=RED, lw=1.6, label='TAIL (bonds + puts + fee)')\n"
            "    ax.fill_between(actual.index, actual.values, bond_part.values, color=RED, alpha=.12)\n"
            "    ax.set_ylabel('growth of $1'); ax.legend()\n"
            "    ax.set_title(f\"The wedge is the insurance bill: {dm['alpha_ann_pct']:.1f}%/yr of pure put-cost\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"insurance bill: {dm['alpha_ann_pct']:.2f}%/yr (monthly alpha vs IEF, HAC t = {dm['t_alpha']:.2f})\")\n"
            "else:\n"
            "    print('cache missing - frozen:', R['alpha_m_ann'], '%/yr, t =', R['alpha_m_t'])"
        ),
        md(
            f"The insurance itself — not the bonds — costs **≈{abs(R['alpha_m_ann']):.0f}%/yr** "
            f"(statistically solid: the quants notebook shows *t* = {R['alpha_m_t']:.1f}, and the bleed "
            "is there both before and after COVID). Of that, only 0.59 points is the management fee; "
            "the rest is the **put ladder burning down** month after month.\n\n"
            "**Who ever came out ahead?** Take every possible monthly entry since launch and hold to "
            "today:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    co = st.cohort_table(TAPE)\n"
            "    tot = co['cohort_returns'] * 100\n"
            "    fig, ax = plt.subplots()\n"
            "    colors = [GREEN if v > 0 else RED for v in tot.values]\n"
            "    ax.bar(tot.index, tot.values, width=22, color=colors)\n"
            "    ax.axhline(0, color=GREY, lw=1)\n"
            "    ax.set_ylabel('total return if held to today (%)'); ax.set_xlabel('month you bought TAIL')\n"
            "    ax.set_title(f\"{co['n_ahead_now']} of {co['n_cohorts']} entry months is above water today (best: +{co['best_pct']:.1f}%)\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"ahead today: {co['n_ahead_now']}/{co['n_cohorts']}  median {co['median_pct']:.1f}%  \"\n"
            "          f\"ever-ahead along the way: {co['share_ever_ahead']:.1f}%\")\n"
            "else:\n"
            "    print('cache missing - frozen:', R['ahead_now'], 'of', R['n_cohorts'], 'ahead; median', R['median'], '%')"
        ),
        md(
            f"A wall of red. **{R['ahead_now']} of {R['n_cohorts']}** entry months is above water today "
            f"— by **+{R['best']:.1f}%**, and that buyer entered in *{R['best_date'][:4]}*, long after "
            f"the crash. The twist: **{R['ever_ahead']:.0f}%** of buyers *were* ahead at some point — "
            "the 2020 spike lifted everyone who bought early — and every single one of them gave it "
            "back within a year. The insurance pays; holding it takes the payment back."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The bleed is genuine and statistically solid: **{R['cagr_pct']:.1f}%/yr** "
            f"overall, **≈{abs(R['alpha_m_ann']):.0f}%/yr** of pure insurance cost once bonds are "
            "stripped out, and the market has priced insurance above what it delivers in 5 months out "
            "of 6 for 33 years. This is the volatility risk premium — from the paying side.\n"
            "- **Tradability — Mirage** (for the buyer). There is nothing to deploy on the long side: "
            "every portfolio blend with TAIL lowered returns by 1–4+ points a year for a modest "
            "crash cushion, and the one jackpot evaporated in 337 days. The tradable side of this "
            "premium is the *seller's* — studies [92](../../92-easy-money/) and "
            "[63](../../63-free-fall/).\n"
            f"- **\"The 2020 win made it worth it\"? — Busted.** At the very top of the payoff, a "
            f"day-one holder was up just **+{R['holder_peak']:.1f}%** — the bleed had already eaten "
            "three years of gains — and 337 days later even that was gone."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Insurance isn't supposed to be an investment.** A negative-carry hedge *can* be "
            "rational if it lets you hold more risk elsewhere — but then the honest accounting is the "
            "blend table (quants notebook §4d): here the crash cushion never repaid its annual bill.\n"
            "- **The premium is the point.** The same gap that bleeds TAIL is what pays the short-vol "
            "carry the desk graded Real in [92-easy-money](../../92-easy-money/) and "
            "[63-free-fall](../../63-free-fall/) — and what the SKEW index *quotes* but doesn't "
            "predict ([86-tail-radar](../../86-tail-radar/)).\n"
            "- **Timing is the only escape hatch** — buy the insurance only when it's cheap and crashes "
            "are near. That requires a crash signal; 86 tested the most famous one and found none.\n\n"
            "*Think you can hold a −7.7%/yr bleed long enough to catch the next 2020 — and sell the "
            "spike within weeks? That's the only version of this trade that has ever worked on paper. "
            "Show it with an honest exit rule — then we'll talk.*"
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
            "# The Price of Crash Insurance — a quantitative teardown 🔬\n"
            "### HAC drift · alpha-vs-collateral decomposition (daily / monthly / sub-periods) · the "
            "2020 episode · buyer-side variance premium from ^VIX+SPY · cohort accounting · blend "
            "costs · a 20-seed fair-vs-planted-bleed control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "— **crash insurance bleeds every year, and the bleed is the volatility risk premium from "
            "the buyer's side** — is *likely true by construction*, so the job is to measure it "
            "honestly on the live product (TAIL) and pin the mechanism.\n\n"
            "> ⚠️ **Data note.** yfinance daily closes: TAIL/IEF/SPY **auto-adjusted total-return**, "
            "^VIX a **level**. TAIL numbers are **net of its 0.59%/yr ER** (inside the NAV). TAIL "
            "buy-and-hold is static — no signal, no lag; the variance-premium series uses strictly "
            "prior-month VIX (one documented one-month lag). Survivorship runs *against* the claim "
            "(TAIL is the surviving tail fund; the dead ones bled faster). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (as-of " + R["asof"] + ", fingerprint `"
            + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | Monthly drift **{R['monthly_bps']:.1f} bps/mo** (HAC "
            f"**t = {R['monthly_t']:.2f}**); monthly alpha vs IEF **{R['alpha_m_ann']:.2f}%/yr** (HAC "
            f"**t = {R['alpha_m_t']:.2f}**), negative in both crash-free sub-periods; buyer-side "
            f"variance premium over {R['vrp_months']} months: RV−IV t = **{R['vrp_var_t']:.2f}** "
            f"(vol-points t = {R['vrp_vol_t']:.2f}), buyer wins {R['vrp_win']:.0f}% of months. |\n"
            f"| **Tradability** | `MIRAGE` | No deployable long-side edge: $1 → $0.50 net; SPY/TAIL "
            f"blends drag CAGR {R['blends'][0][3]:.1f}–{R['blends'][2][3]:.1f} pp/yr at 5 bps; the "
            f"2020 jackpot was given back in {R['giveback_days']} days. The premium's harvestable side "
            "is the seller's (studies 92 / 63). |\n"
            f"| **2020 win kept?** | `BUSTED` | {R['ahead_now']} of {R['n_cohorts']} cohorts above "
            f"water (+{R['best']:.1f}%, a {R['best_date'][:4]} entry); {R['ever_ahead']:.0f}% were "
            f"ever ahead and all gave it back; day-one holder peaked at +{R['holder_peak']:.1f}%. |\n\n"
            "> 💡 In plain words: the bleed is real, measured three independent ways — and it is "
            "exactly the premium the desk's short-vol studies collect."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{TAIL}_t$ be TAIL's daily total return and $r^{IEF}_t$ its collateral proxy's. "
            "The design ($\\approx$ 90% intermediate Treasuries + an OTM SPX put ladder) implies\n\n"
            "$$r^{TAIL}_t = \\alpha + \\beta\\, r^{IEF}_t + \\varepsilon_t,$$\n\n"
            "where $\\alpha$ captures the put sleeve **plus the fee** — the insurance bill, isolated "
            "from duration. The claim decomposes into:\n\n"
            "- **H₁ (the bleed exists).** $\\alpha < 0$ and HAC-significant; raw drift negative.\n"
            "- **H₂ (the bleed is the VRP).** The buyer-side variance premium "
            "$\\mathbb{E}[RV - IV] < 0$ on real ^VIX+SPY, same sign and magnitude class.\n"
            "- **H₃ (the jackpot doesn't redeem it).** Net of 2020, no entry cohort ends ahead.\n\n"
            "We find **all three supported** — H₁ at monthly HAC t = "
            f"{R['alpha_m_t']:.2f}, H₂ at t = {R['vrp_var_t']:.2f} over 33 years, H₃ with "
            f"{R['ahead_now']}/{R['n_cohorts']} cohorts ahead."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The volatility risk premium is the desk's most-replicated finding from the **short** side "
            "([92-easy-money](../../92-easy-money/): short VIXY carry, HAC t = +2.31; "
            "[63-free-fall](../../63-free-fall/): SVXY carry — both `Real`). If a live *long-vol* "
            "product bleeds at a compatible rate, the premium is confirmed **from both ends of the "
            "pipe** — the cleanest possible triangulation, on tradable NAVs rather than model "
            "surfaces. And the buyer's side is where retail actually lives: the 2020 spike chart "
            "sells tail funds; the annual bill is the part the chart doesn't show.\n\n"
            "Inference discipline: HAC/Newey-West t on every serially-correlated mean (plug-in lag "
            "$\\lfloor 4(n/100)^{2/9}\\rfloor$); monthly frequency as the headline for the "
            "decomposition (daily microstructure noise dilutes t); crash-free sub-periods so no leg "
            "owes its sign to the jackpot; the variance series uses **strictly prior** month-end VIX."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance TAIL/IEF/SPY (total-return) + ^VIX (level), TAIL era "
            f"{R['start']} → {R['end']} ({R['n_days']} daily returns, {R['n_months']} complete "
            f"months), as-of {R['asof']}, fingerprint `{R['fingerprint']}`.\n"
            "- **H₁.** HAC t of the raw drift (daily + monthly); OLS of TAIL on IEF with HAC t on "
            "$\\alpha$ (daily, monthly, pre-COVID → 2020-02-19, post-COVID 2020-07-01 →).\n"
            "- **H₂.** Monthly $RV_m = \\sum_d \\ln^2(S_d/S_{d-1})$ vs $IV_m = (VIX_{m-1}/100)^2/12$ "
            "(prior month-end — one clean lag); HAC t of the mean in variance and vol-point units; "
            "1993→ and TAIL-era windows. **Model arithmetic on real tapes, labeled** — no strikes, no "
            "roll mechanics.\n"
            "- **H₃.** Every month-end entry cohort held to as-of; ever-ahead vs ahead-now.\n"
            "- **Costs.** SPY/TAIL blends: monthly rebalance, 5 bps one-way × turnover; TAIL ER inside "
            "NAV.\n"
            "- **Control.** Synthetic bond+put world with a planted net bleed, **20 seeds per "
            "setting**: fair insurance (bleed 0) must not be flagged; a planted 5%/yr bleed must be "
            "recovered in size and t."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The bleed, measured — drift and the alpha-vs-collateral decomposition\n\n"
            "Raw drift first, then the regression that separates the puts from the bonds."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = st.drift_stats(TAPE); dd = st.decompose_vs_ief(TAPE); dm = st.decompose_monthly(TAPE)\n"
            "    sp = st.alpha_subperiods(TAPE)\n"
            "    print(f\"drift   : total {d['total_pct']:+.1f}%  CAGR {d['cagr_pct']:+.2f}%/yr  vol {d['ann_vol_pct']:.1f}%\")\n"
            "    print(f\"          daily {d['daily_mean_bps']:+.2f} bps (HAC t {d['daily_t']:+.2f})   \"\n"
            "          f\"monthly {d['monthly_mean_bps']:+.2f} bps (HAC t {d['monthly_t']:+.2f}, n={d['n_months']})\")\n"
            "    print(f\"alpha   : daily {dd['alpha_ann_pct']:+.2f}%/yr (t {dd['t_alpha']:+.2f}, beta {dd['beta_ief']:.2f}, R2 {dd['r2']:.3f})\")\n"
            "    print(f\"          monthly {dm['alpha_ann_pct']:+.2f}%/yr (t {dm['t_alpha']:+.2f}, beta {dm['beta_ief']:.2f}) <- headline\")\n"
            "    print(f\"          pre-COVID {sp['pre']['alpha_ann_pct']:+.2f}%/yr (t {sp['pre']['t_alpha']:+.2f})   \"\n"
            "          f\"post-COVID {sp['post']['alpha_ann_pct']:+.2f}%/yr (t {sp['post']['t_alpha']:+.2f})\")\n"
            "    labels = ['daily', 'monthly\\n(headline)', 'pre-COVID', 'post-COVID']\n"
            "    tvals = [dd['t_alpha'], dm['t_alpha'], sp['pre']['t_alpha'], sp['post']['t_alpha']]\n"
            "else:\n"
            "    labels = ['daily', 'monthly\\n(headline)', 'pre-COVID', 'post-COVID']\n"
            "    tvals = [R['alpha_d_t'], R['alpha_m_t'], R['pre_t'], R['post_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, RED, AMBER, AMBER], width=.55)\n"
            "ax.axhline(-2, ls='--', c=RED, label='|t| = 2 bar')\n"
            "for i, v in enumerate(tvals): ax.annotate(f't={v:.2f}', (i, v), ha='center', va='top')\n"
            "ax.set_ylabel('HAC t of alpha vs IEF'); ax.legend()\n"
            "ax.set_title('The insurance bill is significant at monthly frequency and negative everywhere')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: strip the bond sleeve out and the puts + fee cost "
            f"**{R['alpha_m_ann']:.1f}%/yr** — significant (**t = {R['alpha_m_t']:.2f}**) at the "
            f"monthly frequency where compounding soaks up daily noise, and negative both **before** "
            f"COVID ({R['pre_ann']:.1f}%/yr, t = {R['pre_t']:.2f}) and **after** it "
            f"({R['post_ann']:.1f}%/yr, t = {R['post_t']:.2f}) — the crash itself excluded from both "
            "legs, so no sub-period owes its sign to the jackpot. The raw monthly drift clears the bar "
            f"on its own (t = {R['monthly_t']:.2f})."
        ),
        md(
            "### 4b · The premium named — RV − IV on real ^VIX + SPY, 1993 →\n\n"
            "The buyer of one month of variance pays $(VIX_{m-1}/100)^2/12$ and receives the month's "
            "realized variance. Cumulate the buyer's P&L:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    vix = TAPE['vix'].dropna(); spy = TAPE['spy'].dropna()\n"
            "    lr = np.log(spy / spy.shift(1)).dropna()\n"
            "    rv = (lr**2).groupby(lr.index.to_period('M')).sum()\n"
            "    iv = ((vix.resample('ME').last()/100.0)**2/12.0); iv.index = iv.index.to_period('M')+1\n"
            "    pnl = (rv - iv).dropna(); pnl = pnl[pnl.index >= pd.Period('1993-02','M')]\n"
            "    cum = pnl.cumsum() * 100\n"
            "    vp = st.variance_premium(TAPE)\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(cum.index.to_timestamp(), cum.values, color=RED, lw=1.5)\n"
            "    ax.axhline(0, color=GREY, lw=1)\n"
            "    ax.set_ylabel('cumulative buyer P&L (variance points x 100)')\n"
            "    ax.set_title(f\"The variance buyer's running tab, 1993-2026: down in {100-vp['share_buyer_wins']:.0f}% of months (HAC t = {vp['t_var']:.2f})\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"RV-IV {vp['mean_var_ann']*100:+.2f} var pts/yr (t {vp['t_var']:+.2f})  vol-pts {vp['mean_vol_pts']:+.2f} (t {vp['t_vol']:+.2f})\")\n"
            "    print(f\"implied {vp['iv_ann_vol']:.2f}% vs realized {vp['rv_ann_vol']:.2f}%  buyer wins {vp['share_buyer_wins']:.1f}% of {vp['n_months']} months\")\n"
            "else:\n"
            "    print('cache missing - frozen:', R['vrp_var'], 'var pts (t', R['vrp_var_t'], '), buyer wins', R['vrp_win'], '%')"
        ),
        md(
            f"> 💡 In plain words: for 33 years the market has quoted insurance at **{R['iv']:.1f}%** "
            f"vol and delivered **{R['rv']:.1f}%** — the buyer pays the gap, loses in **5 months out "
            f"of 6**, and the mean is solidly nonzero (t = {R['vrp_var_t']:.2f} in variance units, "
            f"{R['vrp_vol_t']:.2f} in vol points). Within the shorter TAIL era the variance-unit t "
            f"drops below 2 ({R['vrp_era_var_t']:.2f} — nine years is short for a crash-skewed "
            f"series) while vol-points stay clear ({R['vrp_era_vol_t']:.2f}); the 33-year window is "
            "the honest estimate of the premium itself. *(Model arithmetic on real tapes — labeled; "
            "no strikes, no roll mechanics. TAIL's alpha is the live-product evidence.)*"
        ),
        md(
            "### 4c · The 2020 episode + cohort accounting — H₃\n\n"
            "The jackpot, end-to-end, and every month-end buyer's final ledger."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cv = st.covid_episode(TAPE); co = st.cohort_table(TAPE)\n"
            "    print(f\"2020: TAIL +{cv['tail_gain_pct']:.2f}% (to {cv['tail_peak_date']}) while SPY {cv['spy_fall_pct']:.2f}%\")\n"
            "    print(f\"      given back by {cv['giveback_date']} ({cv['giveback_days']} days); day-one holder \"\n"
            "          f\"{cv['holder_at_peak_pct']:+.2f}% at the peak -> {cv['holder_now_pct']:+.2f}% now\")\n"
            "    print(f\"cohorts: {co['n_ahead_now']}/{co['n_cohorts']} ahead now (best {co['best_pct']:+.2f}% entered {co['best_date']}); \"\n"
            "          f\"median {co['median_pct']:.1f}%; ever-ahead {co['share_ever_ahead']:.1f}%\")\n"
            "    tot = co['cohort_returns'] * 100\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.bar(tot.index, tot.values, width=22, color=[GREEN if v > 0 else RED for v in tot.values])\n"
            "    ax.axhline(0, color=GREY, lw=1)\n"
            "    ax.set_ylabel('hold-to-as-of total return (%)'); ax.set_xlabel('entry month')\n"
            "    ax.set_title(f\"Every entry cohort, held to {R['end']}: {co['n_ahead_now']} of {co['n_cohorts']} above water\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing - frozen: 1/110 ahead, median', R['median'], '%, ever-ahead', R['ever_ahead'], '%')"
        ),
        md(
            f"> 💡 In plain words: the insurance **worked** — +{R['covid_gain']:.0f}% while the market "
            f"fell a third — and it still didn't pay. The bleed had eaten so much that a day-one "
            f"holder was up only **+{R['holder_peak']:.1f}%** at the very top, and "
            f"**{R['giveback_days']} days** later the whole spike was gone. "
            f"{R['ever_ahead']:.0f}% of cohorts were ahead at *some* month-end; **"
            f"{R['ahead_now']} of {R['n_cohorts']}** is ahead today, by +{R['best']:.1f}%, from a "
            f"{R['best_date'][:4]} entry that never saw the crash. Net of the 2020 win: nobody."
        ),
        md(
            "### 4d · Tradability — the insurance inside a portfolio\n\n"
            "The steelman for a permanent negative-carry hedge is portfolio-level: does the crash "
            "cushion buy back its bill? Monthly-rebalanced SPY/TAIL blends, 5 bps one-way × turnover:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.hedged_portfolio(TAPE, w_tail=w, cost_bps=5.0) for w in (0.05, 0.10, 0.20)]\n"
            "    blends = [(int(r['w_tail']*100), r['port_cagr_pct'], r['spy_cagr_pct'], r['cagr_drag_pct'],\n"
            "               r['port_mdd_pct'], r['spy_mdd_pct'], r['port_covid_dd_pct'], r['spy_covid_dd_pct']) for r in rows]\n"
            "else:\n"
            "    blends = R['blends']\n"
            "x = np.arange(3); drag = [b[3] for b in blends]; cushion = [b[7]-b[6] for b in blends]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x-.18, drag, .34, color=RED, label='CAGR given up EVERY year (pp)')\n"
            "ax.bar(x+.18, cushion, .34, color=GREEN, label='COVID drawdown trimmed, ONCE (pp)')\n"
            "for i in x:\n"
            "    ax.annotate(f'-{drag[i]:.1f}', (i-.18, drag[i]), ha='center', va='bottom')\n"
            "    ax.annotate(f'+{cushion[i]:.1f}', (i+.18, cushion[i]), ha='center', va='bottom')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{100-b[0]}/{b[0]} SPY/TAIL' for b in blends])\n"
            "ax.set_ylabel('percentage points'); ax.legend()\n"
            "ax.set_title('The annual bill vs the one-off cushion (5 bps one-way x turnover)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for b in blends: print(f'{100-b[0]}/{b[0]}: CAGR {b[1]:+.2f}% vs SPY {b[2]:+.2f}% (drag {b[3]:.2f} pp/yr) | '\n"
            "                       f'maxDD {b[4]:.1f}% vs {b[5]:.1f}% | COVID DD {b[6]:.1f}% vs {b[7]:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: a 90/10 blend gave up **{R['blends'][1][3]:.1f} pp of CAGR every "
            f"year** to trim the COVID drawdown by **{R['blends'][1][7]-R['blends'][1][6]:.1f} pp "
            f"once** (and max drawdown by {R['blends'][1][5]-R['blends'][1][4]:.1f} pp). Every blend "
            "ends behind 100% SPY over the fund's life. The long side has no deployable edge — "
            "**Mirage** — while the *short* side of the identical premium carried the desk's Real "
            "stamps in 92/63 (themselves only Fragile, because crash risk cuts the other way)."
        ),
        md(
            "### 4e · Faithful-engine control — fair insurance vs a planted bleed (20 seeds)\n\n"
            "Synthetic bond+put world. With `bleed = 0` the puts are actuarially **fair** (expected "
            "jump payoffs exactly repay the premium): the alpha detector must stay quiet. A planted "
            "5%/yr bleed must be recovered in size and flagged in every seed. Never cited for a stamp."
        ),
        code(
            "rows = st.control_summary(data.synthetic_world, bleeds=(0.0, 0.05), n_seeds=20)\n"
            "for r in rows:\n"
            "    print(f\"planted bleed {r['bleed_pct']:.0f}%/yr: mean alpha {r['mean_alpha_ann_pct']:+.2f}%/yr  \"\n"
            "          f\"mean HAC t {r['mean_t']:+.2f}  flagged {r['share_flagged']:.0f}% of {r['n_seeds']} seeds\")\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "labs = [f\"fair insurance\\n(bleed 0)\", f\"planted bleed\\n5%/yr\"]\n"
            "ax.bar(labs, [r['mean_t'] for r in rows], color=[GREY, RED], width=.5)\n"
            "ax.axhline(-2, ls='--', c=RED, label='|t| = 2 bar')\n"
            "for i, r in enumerate(rows): ax.annotate(f\"t={r['mean_t']:.2f}\", (i, r['mean_t']), ha='center', va='top')\n"
            "ax.set_ylabel('mean HAC t of alpha (20 seeds)'); ax.legend()\n"
            "ax.set_title('Fair insurance is not flagged; a planted bleed lights up')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: when insurance is priced **fairly** the engine finds nothing "
            f"(mean t = {R['ctrl'][0][2]:.2f}, {R['ctrl'][0][3]:.0f}% flagged — the nominal false-"
            f"positive rate); a planted {R['ctrl'][1][0]:.0f}%/yr bleed is recovered at "
            f"{R['ctrl'][1][1]:.2f}%/yr with mean t = {R['ctrl'][1][2]:.2f}, flagged in "
            f"{R['ctrl'][1][3]:.0f}% of seeds. The real-tape −7.7%/yr at t = {R['alpha_m_t']:.2f} is "
            "a measurement, not an artefact. *(Machinery proof only.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — three independent clears on the real tape: monthly drift "
            f"**{R['monthly_bps']:.1f} bps/mo** (HAC t = {R['monthly_t']:.2f}); monthly alpha vs its "
            f"own collateral **{R['alpha_m_ann']:.2f}%/yr** (HAC **t = {R['alpha_m_t']:.2f}**, "
            f"negative in both crash-free sub-periods); buyer-side variance premium "
            f"**{R['vrp_var']:.2f} var pts/yr** over {R['vrp_months']} months (HAC "
            f"**t = {R['vrp_var_t']:.2f}**; vol-points t = {R['vrp_vol_t']:.2f}). Survivorship runs "
            "against the claim and it clears anyway.\n"
            f"- **Tradability `MIRAGE`** — for the buyer: $1 → $0.50 net of fee; blends drag "
            f"{R['blends'][0][3]:.1f}–{R['blends'][2][3]:.1f} pp/yr at 5 bps for a one-off cushion; "
            f"the jackpot lasted {R['giveback_days']} days. The harvestable side is the seller's "
            "([92](../../92-easy-money/), [63](../../63-free-fall/) — Real / Fragile).\n"
            f"- **2020 win kept? `BUSTED`** — {R['ahead_now']} of {R['n_cohorts']} cohorts ahead "
            f"(+{R['best']:.1f}%, {R['best_date'][:4]} entry); day-one holder peaked at "
            f"+{R['holder_peak']:.1f}%; {R['ever_ahead']:.0f}% were ahead mid-flight and all gave it "
            "back."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Both ends of the pipe now agree.** Short side: [92-easy-money](../../92-easy-money/) "
            "(+14.8%/yr shorting VIXY, HAC t = +2.31) and [63-free-fall](../../63-free-fall/) "
            "(SVXY +11.8%/yr). Long side (here): −7.7%/yr alpha on the live long-vol fund. One "
            "premium, two NAVs, opposite signs — the volatility risk premium triangulated on "
            "tradable products.\n"
            "- **The signal axis is separate.** [86-tail-radar](../../86-tail-radar/) showed the "
            "*price* of tails (SKEW) does not predict crashes — so \"buy insurance only when it's "
            "about to pay\" has no documented trigger. Without timing, the carry math here is the "
            "whole story.\n"
            "- **The honest use-case.** A permanent tail sleeve is a *utility* purchase (sleep, "
            "mandate constraints, leverage room), not an investment: budget it like an expense — "
            "≈7.7%/yr of the sleeve — and expect the cohort table, not the 2020 poster.\n\n"
            "*The reproducible core is offline and deterministic; buy-and-hold is static (no lag), the "
            "variance series uses strictly prior-month VIX (one documented lag). Methods and sources: "
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
