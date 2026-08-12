"""Generate the two narrative notebooks for Study 887 (High-Yield Muni Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF
panels under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return
# net-of-fee ETF tape, as-of 2026-06-30, fingerprint db0172501766).
R = dict(
    asof="2026-06-30", fingerprint="db0172501766",
    start="2009-03", end="2026-06", n_months=208, years=17.3, rate=40.8,
    spread_bps=20.21, spread_t=1.80, spread_pp=2.43,
    boot_lo=-1.44, boot_hi=42.23, boot_fracneg=0.036,
    ann=dict(HYD=5.38, MUB=3.20, TFI=3.01, HYG=7.26, BIL=1.25),
    sharpe=dict(HYD=0.478, MUB=0.408, TFI=0.341, HYG=0.732),
    sharpe_adv=0.069,
    # eras: (label, bps/mo, HAC t, n)
    eras=[("2009-2016", 27.4, 1.85, 94), ("2017-2026", 14.3, 0.91, 114),
          ("2020 COVID", -18.4, -0.15, 12), ("2022 rate shock", -78.0, -4.35, 12)],
    # sharpe advantage by half: (label, HYD, MUB, diff)
    halves=[("2009-2017", 1.00, 0.87, 0.13), ("2018-2026", 0.11, -0.04, 0.15)],
    income=dict(HYD=5.43, MUB=2.71, HYG=6.04),
    tey=9.17,
    aftertax_ann=dict(HYD=5.38, MUB=3.20, HYG=4.67),
    aftertax_sharpe=dict(HYD=0.53, MUB=0.51, HYG=0.50),
    dd=dict(HYD=-35.6, MUB=-13.7, HYG=-22.0),
    dd_dates=dict(HYD="2020-02-26 - 2020-03-18", MUB="2020-03-09 - 2020-03-19",
                  HYG="2020-02-13 - 2020-03-23"),
    # costs: (one-way bps, drag bps/yr, net bps/yr)
    costs=[(5.0, 0.6, 242), (15.0, 1.7, 241), (30.0, 3.5, 239)], gross_bps=243,
    # synthetic: (planted %/yr, mean bps, HAC t, CI lo, CI hi, frac<0)
    syn=[(0.0, 5.18, 1.13, -3.5, 14.1, 0.128), (3.0, 30.18, 6.59, 21.5, 39.1, 0.000)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Credit + tax wrapper?: Half true](https://img.shields.io/badge/Credit_%2B_tax_wrapper%3F-Half_true-8b949e?style=flat-square)\n\n"
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

from hy_muni import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_prices()                       # total-return closes, frozen as-of
    PR = data.load_price_only()                   # price-only closes (income leg)
    M = st.monthly_returns(PX, asof=data.AS_OF)
    INC = st.monthly_income(PX, PR, asof=data.AS_OF)
    MC = st.align_common(M, ["HYD", "MUB", "TFI", "HYG", "BIL"])   # HYD era
    RATE = data.TOP_MARGINAL_RATE
else:
    PX = PR = M = INC = MC = None
    RATE = 0.408
print("real ETF cache present:", HAVE_REAL,
      "| HYD-era months:", (0 if MC is None else len(MC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# High-yield munis: a 9% yield hiding behind a tax break 🏛️\n"
            "### The high-yield muni credit premium — tested live, in plain English\n\n"
            + BADGES +
            "A *municipal bond* is a loan to a city, school district or toll road. Its magic trick: "
            "the interest is **exempt from federal income tax**. A *high-yield* muni is one from a "
            "shakier issuer, so it pays a fatter coupon. Put those together and you get a fund (HYD) "
            "advertising a yield that looks small on paper — until you remember a top-bracket investor "
            "keeps **all** of it while paying tax on ordinary junk bonds.\n\n"
            "The pitch has two parts, and they are **not** equally true:\n\n"
            "1. **The tax wrapper** — is a high-yield muni's *tax-equivalent* yield really higher than "
            "taxable junk? (Spoiler: **yes, and it's arithmetic**.)\n"
            "2. **The credit premium** — does high-yield muni actually *out-earn* plain investment-"
            "grade muni for the extra risk, reliably? (Spoiler: **thinly, and it falls apart in a "
            "crisis**.)\n\n"
            "We read the live tape: real, fee-charging ETFs since 2009.\n\n"
            "> 📓 **Plain-language layer.** Want the regressions and robustness checks? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it, from **total-return, net-of-fee** ETF "
            "prices. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the tax-free yield really worth more than taxable junk? | **Yes — mechanically.** "
            f"HYD's **tax-equivalent yield is ~{R['tey']:.1f}%** vs **{R['income']['HYG']:.1f}%** for "
            "taxable HY, and its *after-tax* return actually beats taxable junk "
            f"(**{R['aftertax_ann']['HYD']:.1f}%** vs **{R['aftertax_ann']['HYG']:.1f}%**/yr). |\n"
            "| Does HY-muni beat plain (safe) muni for the extra risk? | **Only weakly.** "
            f"**+{R['spread_pp']:.1f}%/yr** on average — but the statistics are borderline (below our "
            "bar), and it turns sharply *negative* exactly when markets seize up. |\n"
            "| Can you collect it cheaply? | **The trade is cheap** (one fund swap), but the *edge* is "
            "thin and fragile, so it isn't a green light. |\n"
            "| What's the catch? | **The crash.** In March 2020 HYD fell "
            f"**{R['dd']['HYD']:.0f}%** while safe muni fell only **{R['dd']['MUB']:.0f}%**; in 2022 "
            "HY-muni *lost* to safe muni by nearly **1% per month**. |"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"High-yield munis pay a fat spread over investment-grade munis — a credit premium — "
            "and because the coupons are federally tax-free, a top-bracket investor earns a "
            "tax-equivalent yield that buries taxable junk bonds.\"*\n\n"
            "Two mechanisms are bundled here. One is **tax law** (muni coupons are exempt — not an "
            "opinion). The other is a **risk premium** (you get paid for lending to shakier issuers in "
            "a thin, illiquid market). Bundling them is how a marketing deck turns a borderline credit "
            "bet into a headline yield. Our job is to un-bundle them."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If both legs held, a top-bracket investor would have a rare free lunch: a high tax-"
            "equivalent yield *and* a paid credit premium, collectible in one ETF. If only the **tax** "
            "leg holds, HY-muni is still a smart place to park taxable-account money — but you're "
            "buying a *tax break*, not a *credit edge*, and you must respect the crash. Telling the two "
            "apart is the whole point."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"Take every complete month since the HY-muni ETF listed (**{R['start']} → {R['end']}, "
            f"{R['n_months']} months**) and:\n\n"
            "1. **Measure the credit spread** — HYD minus MUB (safe muni), with statistics that aren't "
            "fooled by streaky months, and a bootstrap confidence band.\n"
            "2. **Check the crises** — 2020 and 2022, where muni illiquidity is supposed to bite.\n"
            "3. **Do the tax math** — back out each fund's coupon, gross up the muni's tax-free yield, "
            "and race the *after-tax* returns against taxable junk (HYG).\n"
            "4. **Price the pain** — the drawdown, and the cost of the trade."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Growth of $100** in high-yield muni vs plain investment-grade muni, since both traded."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = PX[[\"HYD\", \"MUB\"]].loc[\"2009-03-01\":].dropna()\n"
            "    g = 100.0 * px / px.iloc[0]\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(g.index, g[\"HYD\"], c=GREEN, lw=1.8, label=\"HYD - high-yield muni\")\n"
            "    ax.plot(g.index, g[\"MUB\"], c=GREY, lw=1.8, label=\"MUB - investment-grade muni\")\n"
            "    ax.set_ylabel(\"growth of $100 (total return, net of fees)\")\n"
            "    ax.set_title(\"17 years live: HY-muni out-earns safe muni - but watch the dips\")\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final: HYD ${g[\"HYD\"].iloc[-1]:.0f}  vs  MUB ${g[\"MUB\"].iloc[-1]:.0f}')\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', R['ann'])"
        ),
        md(
            f"HY-muni compounds at **{R['ann']['HYD']:.2f}%/yr** vs **{R['ann']['MUB']:.2f}%/yr** for "
            f"safe muni — a gap of **+{R['spread_pp']:.1f} pp/yr** for {R['years']:.0f} years. But the "
            f"quants notebook shows the gap is **statistically thin** (HAC *t* = {R['spread_t']:.2f}, "
            "*below* the desk's *t* ≥ 2 bar) and its bootstrap confidence band touches zero. Real "
            "direction, weak certainty."
        ),
        md(
            "**The tax break — where the yield actually comes from.** Let's back out each fund's coupon "
            "(income) and gross up the muni's tax-free yield to a *tax-equivalent* yield a top-bracket "
            "investor would compare against taxable junk."
        ),
        code(
            "if HAVE_REAL:\n"
            "    iy = st.income_yields(INC.loc[MC.index], ['HYD', 'MUB', 'HYG'])\n"
            "    tey = st.tax_equivalent_yield(iy['HYD'], RATE)\n"
            "    hyg_y = iy['HYG']\n"
            "else:\n"
            "    iy = R['income']; tey = R['tey']; hyg_y = R['income']['HYG']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "bars = ['HYD raw\\n(tax-free)', f'HYD tax-equivalent\\n@ {RATE*100:.0f}%', 'HYG\\n(taxable junk)']\n"
            "vals = [iy['HYD'], tey, hyg_y]\n"
            "ax.bar(bars, vals, color=[GREY, GREEN, RED], width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(hyg_y, ls='--', c=RED, alpha=.5)\n"
            "ax.set_ylabel('yield (% per year)')\n"
            "ax.set_title('The tax break flips the ranking: HY-muni beats taxable junk after tax')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HYD raw {iy[\"HYD\"]:.2f}%  ->  tax-equivalent {tey:.2f}%   vs  HYG taxable {hyg_y:.2f}%')"
        ),
        md(
            f"On paper taxable junk (HYG) yields **{R['income']['HYG']:.1f}%** and HY-muni only "
            f"**{R['income']['HYD']:.1f}%** — junk looks better. But once you count the tax break, "
            f"HY-muni's **tax-equivalent yield is ~{R['tey']:.1f}%**. That's the whole trick, and it's "
            "just arithmetic: `yield / (1 − tax rate)`. This part of the claim is **real and "
            "mechanical**."
        ),
        md(
            "**The catch — the crash you carry.** The thin muni-junk market gaps hard when liquidity "
            "vanishes. Here is the drawdown bill around COVID."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = PX[[\"HYD\", \"MUB\"]].loc[\"2019-06-01\":\"2020-12-31\"].dropna()\n"
            "    dd = px / px.cummax() - 1.0\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(dd.index, dd[\"HYD\"]*100, c=RED, lw=1.4, label=\"HYD drawdown\")\n"
            "    ax.plot(dd.index, dd[\"MUB\"]*100, c=GREY, lw=1.4, label=\"MUB drawdown\")\n"
            "    ax.set_ylabel(\"drawdown from peak (%)\")\n"
            "    ax.set_title(\"The bill: HY-muni crashed ~36% while safe muni fell ~14% (Mar 2020)\")\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'max DD in window: HYD {dd[\"HYD\"].min()*100:.1f}%  vs  MUB {dd[\"MUB\"].min()*100:.1f}%')\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', R['dd'])"
        ),
        md(
            f"In March 2020 HY-muni fell **{R['dd']['HYD']:.0f}%** vs **{R['dd']['MUB']:.0f}%** for "
            "safe muni — and in 2022 the credit spread went *negative* by nearly 1%/month. That's the "
            "price of admission: when everyone wants out of the thin muni-junk market at once, you're "
            "the one holding it. The premium is your pay for that job — and it isn't reliable."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** +{R['spread_pp']:.1f}%/yr over safe muni for {R['years']:.0f} live "
            f"years, but statistically thin (*t* = {R['spread_t']:.2f}, below the bar; confidence band "
            "touches zero) and it inverts hard in crises.\n"
            "- **Tradability — Fragile.** The trade is cheap (one fund swap) and the **tax wrapper is "
            f"genuinely valuable** (tax-equivalent yield ~{R['tey']:.1f}% vs {R['income']['HYG']:.1f}%; "
            f"after-tax {R['aftertax_ann']['HYD']:.1f}% vs {R['aftertax_ann']['HYG']:.1f}%) — but the "
            "credit edge is thin, crisis-fragile, and only helps a top-bracket taxable investor.\n"
            "- **\"Credit premium + tax wrapper\"? — Half true.** The tax wrapper is real. The credit "
            "premium over safe muni doesn't reliably clear the bar.\n\n"
            "Honest bottom line: buy high-yield muni for the **tax-equivalent yield** if you're "
            "top-bracket and taxable — not for a dependable, all-weather credit edge over safe munis."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why is the credit premium so thin here?** Munis default *rarely*, so most of the "
            "spread is a **liquidity** premium — which is exactly the thing that vanishes (and "
            "reverses) in a panic. You're paid in calm and taxed in crisis.\n"
            "- **Why does the tax leg still matter?** Because it's law, not luck. As long as muni "
            "coupons are federally exempt and you're in a high bracket in a taxable account, the "
            "tax-equivalent-yield gap is real every single month.\n"
            "- **Related desks:** [Study 610 — Fallen-Angels](../../610-fallen-angels-premium/) is the "
            "*taxable* cousin (a within-junk selection premium that **does** clear); [Study 576 — "
            "Muni-Treasury-Ratio](../../576-muni-treasury-ratio/) and [Study 616 — Muni-CEF-Tax-"
            "Loss](../../616-muni-cef-tax-loss/) test other muni folklore.\n\n"
            "*Think the credit premium is only crushed by the 2022 shock? Re-run the era cut without "
            "2022 in the quants notebook and see how much of the thinness is that one year.*"
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
            "# The High-Yield Muni premium — a quantitative teardown 🔬\n"
            "### HAC spread + bootstrap CI · excess Sharpe race · era/crisis cut · tax-equivalent "
            "yield & after-tax race · drawdowns & one-switch costs · synthetic faithful-engine "
            "control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "HY-muni credit premium is graded here **on the live vehicles only**: HYD vs MUB/TFI "
            "2009→2026, HYG as the *taxable-HY* yardstick for the tax comparison, BIL as the tradable "
            "risk-free. The tax-equivalent leg is arithmetic; the credit leg is the one on trial.\n\n"
            "> ⚠️ **Data note.** yfinance auto-adjusted closes = **total-return, net-of-fee** ETF tape "
            "(HYD ER 0.35%, MUB 0.05%, TFI 0.23%, HYG 0.49% already inside); a second price-only pull "
            "backs out the income (coupon) leg. As-of **" + R["asof"] + "** (last complete month), "
            "fingerprint `" + R["fingerprint"] + "`. Tax convention: top-bracket **"
            + f"{R['rate']:.1f}%" + "** (37% + 3.8% NIIT), muni coupons exempt. Numbers in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | HYD − MUB = **+{R['spread_bps']:.2f} bps/mo** (HAC "
            f"**t = {R['spread_t']:.2f}**, n = {R['n_months']}); bootstrap 95% CI **[{R['boot_lo']:.2f}, "
            f"{R['boot_hi']:.2f}]** touches zero; excess-Sharpe edge over MUB only **+{R['sharpe_adv']:.2f}**; "
            "2022 spread **−78 bps/mo (t = −4.35)**. |\n"
            f"| **Tradability** | `FRAGILE` | One switch, drag ≤ {R['costs'][2][1]:.1f} bps/yr vs "
            f"**+{R['costs'][2][2]} bps/yr net** (not a Mirage); but thin Signal + crisis-fragile "
            f"(DD **{R['dd']['HYD']:.1f}% vs {R['dd']['MUB']:.1f}%**) + tax pickup only for top-bracket "
            "taxable accounts. |\n"
            f"| **Credit + tax wrapper?** | `HALF TRUE` | Wrapper real: TEY **{R['tey']:.2f}%** vs HYG "
            f"**{R['income']['HYG']:.2f}%**, after-tax HYD **{R['aftertax_ann']['HYD']:.2f}%** > HYG "
            f"**{R['aftertax_ann']['HYG']:.2f}%**. Credit premium won't certify. |\n\n"
            "> 💡 In plain words: the tax break is a real, mechanical yield pickup; the credit premium "
            "over investment-grade muni is directionally positive but statistically thin and blows up "
            "in stress — so the honest stamps are Weak / Fragile, not green."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "HY munis carry (a) a **credit/liquidity spread** over IG munis and (b) a **tax exemption** "
            "on coupon income. For a top-bracket investor the second is exact; the first is the "
            "empirical question. With monthly total returns and the T-bill (BIL) as the risk-free:\n\n"
            "$$\\text{spread}_t = r_{HYD,t} - r_{MUB,t}, \\qquad "
            "\\text{TEY}_{HYD} = \\frac{y^{coupon}_{HYD}}{1-\\tau}$$\n\n"
            "- **H₁ (credit premium exists).** Mean(HYD − MUB) > 0 with HAC t ≥ 2, a bootstrap CI clear "
            "of zero, holding across sub-eras.\n"
            "- **H₂ (tax wrapper wins).** TEY(HYD) > y(HYG) and after-tax HYD ≥ after-tax HYG.\n"
            "- **H₃ (bankable).** One-switch cost drag ≪ net spread; the crisis drawdown is survivable.\n\n"
            "We find **H₂ clearly supported** (arithmetic), **H₁ only weakly** (t = 1.80, CI to zero, "
            "2022 inversion), so H₃ lands on **Fragile**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The dangerous failure mode is **selling a liquidity premium as a free lunch**. Muni "
            "defaults are rare, so the HY-muni spread is mostly compensation for *illiquidity* — a "
            "premium that is positive in calm and sharply negative in a scramble (Schwert 2017 "
            "decomposes exactly this). If we certified it as `Real/Investable` on the calm-period "
            "average, we'd be handing out a stamp the 2020/2022 tape rips up. The tax leg is different: "
            "it is contractual, so it earns credit even though it is not, by itself, a *risk-adjusted "
            "outperformance*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance total-return net-of-fee closes; HYD-era common sample "
            f"**{R['start']} → {R['end']}** ({R['n_months']} months); as-of {R['asof']}; partial "
            "months dropped; fingerprint `" + R["fingerprint"] + "`.\n"
            "- **H₁.** HAC (Newey-West, Bartlett, rule-of-thumb lags) t on the monthly HYD − MUB "
            "spread + a circular-block bootstrap mean CI; excess-vs-excess Sharpe race on BIL; an era "
            "cut and the 2020/2022 crisis windows.\n"
            "- **H₂.** Income leg = total-return − price-return; tax-equivalent yield y/(1−τ); after-tax "
            "total-return and Sharpe race for a top-bracket investor (muni coupon exempt, HYG coupon "
            "taxed, cash leg taxed).\n"
            "- **H₃.** One-switch cost drag (2 legs × one-way bps over the horizon); daily max-DD.\n"
            "- **Machinery proof.** Synthetic world HYD = premium + 1.15·MUB + ε with a planted knob: "
            "null must stay under t = 2 with a CI straddling zero; planted +3%/yr must light up. Never "
            "market evidence."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · H₁ — the credit spread, its HAC t and bootstrap CI\n\n"
            "Cumulative relative strength (HYD/MUB) plus the monthly spread's HAC test and a "
            "block-bootstrap mean interval."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.premium_series(MC)\n"
            "    h = st.hac_mean(sp.values)\n"
            "    boot = st.bootstrap_mean_ci(sp.values)\n"
            "    rel = (1 + MC[['HYD','MUB']]).cumprod()\n"
            "    ratio = rel['HYD'] / rel['MUB']\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(ratio.index, ratio, c=GREEN, lw=1.6)\n"
            "    ax.axhline(1.0, c=GREY, ls='--', lw=1)\n"
            "    ax.set_yscale('log'); ax.set_ylabel('HYD / MUB cumulative ratio (log)')\n"
            "    ax.set_title(f\"Relative strength: +{h['mean_bps']:.2f} bps/mo, HAC t = {h['tstat']:.2f} (below t=2)\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"HYD - MUB: {h['mean_bps']:+.2f} bps/mo  HAC t = {h['tstat']:+.2f}  (n={h['n']}, lags={h['lags']})\")\n"
            "    print(f\"bootstrap 95% mean CI: [{boot['ci_low_bps']:+.2f}, {boot['ci_high_bps']:+.2f}] bps/mo  frac<0={boot['frac_negative']:.3f}\")\n"
            "    print(f\"excess Sharpe: HYD {st.sharpe_excess(MC,'HYD'):.3f}  MUB {st.sharpe_excess(MC,'MUB'):.3f}  \"\n"
            "          f\"TFI {st.sharpe_excess(MC,'TFI'):.3f}  HYG {st.sharpe_excess(MC,'HYG'):.3f}\")\n"
            "else:\n"
            "    print('cache missing - frozen:', R['spread_bps'], R['spread_t'], (R['boot_lo'], R['boot_hi']), R['sharpe'])"
        ),
        md(
            f"> 💡 In plain words: **+{R['spread_bps']:.2f} bps/mo** for {R['years']:.1f} years, but at "
            f"**HAC t = {R['spread_t']:.2f}** it is *below* the desk's t ≥ 2 bar, and the bootstrap 95% "
            f"CI **[{R['boot_lo']:.2f}, {R['boot_hi']:.2f}]** includes zero. The excess-Sharpe race "
            f"gives HYD **{R['sharpe']['HYD']:.2f}** vs MUB **{R['sharpe']['MUB']:.2f}** — a thin "
            f"**+{R['sharpe_adv']:.2f}** advantage. Directionally right, not robust."
        ),
        md(
            "### 4b · The era cut & the crises — where illiquidity bites\n\n"
            "Neither half certifies alone, and the premium *inverts* in the 2020/2022 stress windows."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.premium_series(MC)\n"
            "    eras = st.era_table(sp, [('2009-03','2016-12','2009-2016'),('2017-01','2026-06','2017-2026'),\n"
            "                             ('2020-01','2020-12','2020 COVID'),('2022-01','2022-12','2022 rate shock')])\n"
            "    rows = [(e['label'], e['mean_bps'], e['tstat']) for e in eras]\n"
            "else:\n"
            "    rows = [(e[0], e[1], e[2]) for e in R['eras']]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [GREEN if v[2] >= 2 else (RED if v[1] < 0 else AMBER) for v in rows]\n"
            "ax.bar([r[0] for r in rows], [r[1] for r in rows], color=cols, width=.6)\n"
            "for i, r in enumerate(rows): ax.annotate(f'{r[1]:+.0f}\\nt={r[2]:.2f}', (i, r[1]), ha='center', va='bottom' if r[1]>=0 else 'top', fontsize=8)\n"
            "ax.axhline(0, c=GREY, lw=1)\n"
            "ax.set_ylabel('HYD - MUB spread (bps/month)')\n"
            "ax.set_title('Thin in each half; deeply NEGATIVE in the 2020/2022 crises')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lab, m_, t in rows: print(f'{lab:16s} {m_:+7.1f} bps/mo  HAC t {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: 2009-2016 was the strongest stretch (+{R['eras'][0][1]:.0f} bps/mo, "
            f"t = {R['eras'][0][2]:.2f}) and 2017-2026 far weaker (t = {R['eras'][1][2]:.2f}); **neither "
            f"clears t = 2**. And in the crises the spread is **{R['eras'][2][1]:.0f} bps/mo** (2020) "
            f"and **{R['eras'][3][1]:.0f} bps/mo at t = {R['eras'][3][2]:.2f}** (2022) — HY-muni *lost* "
            "to safe muni precisely when liquidity evaporated. This is the illiquidity premium showing "
            "its teeth."
        ),
        md(
            "### 4c · H₂ — the tax wrapper (the part that works)\n\n"
            "Tax-equivalent yield and the after-tax total-return race for a top-bracket investor."
        ),
        code(
            "if HAVE_REAL:\n"
            "    iy = st.income_yields(INC.loc[MC.index], ['HYD','MUB','HYG'])\n"
            "    tey = st.tax_equivalent_yield(iy['HYD'], RATE)\n"
            "    at_hyd = st.after_tax_returns(MC, INC.loc[MC.index], 'HYD', RATE, tax_exempt=True)\n"
            "    at_mub = st.after_tax_returns(MC, INC.loc[MC.index], 'MUB', RATE, tax_exempt=True)\n"
            "    at_hyg = st.after_tax_returns(MC, INC.loc[MC.index], 'HYG', RATE, tax_exempt=False)\n"
            "    at_ann = {'HYD': st.ann_return(at_hyd), 'MUB': st.ann_return(at_mub), 'HYG': st.ann_return(at_hyg)}\n"
            "    at_sh = {'HYD': st.after_tax_sharpe(at_hyd, MC, RATE), 'MUB': st.after_tax_sharpe(at_mub, MC, RATE),\n"
            "             'HYG': st.after_tax_sharpe(at_hyg, MC, RATE)}\n"
            "else:\n"
            "    iy = R['income']; tey = R['tey']; at_ann = R['aftertax_ann']; at_sh = R['aftertax_sharpe']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "a1.bar(['HYD raw','HYD TEY','HYG'], [iy['HYD'], tey, iy['HYG']], color=[GREY, GREEN, RED], width=.6)\n"
            "for i, v in enumerate([iy['HYD'], tey, iy['HYG']]): a1.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('yield (%/yr)'); a1.set_title(f'Tax-equivalent yield @ {RATE*100:.0f}%: HY-muni beats taxable junk')\n"
            "labs = ['HYD','HYG','MUB']\n"
            "a2.bar(labs, [at_ann[k] for k in labs], color=[GREEN, RED, GREY], width=.6)\n"
            "for i, k in enumerate(labs): a2.annotate(f\"{at_ann[k]:.1f}%\\nSh {at_sh[k]:.2f}\", (i, at_ann[k]), ha='center', va='bottom', fontsize=9)\n"
            "a2.set_ylabel('after-tax annualised return (%)'); a2.set_title('After-tax race (top bracket)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"income yield: HYD {iy['HYD']:.2f}%  MUB {iy['MUB']:.2f}%  HYG {iy['HYG']:.2f}%\")\n"
            "print(f\"HYD TEY {tey:.2f}%  vs HYG {iy['HYG']:.2f}%\")\n"
            "print(f\"after-tax ann: HYD {at_ann['HYD']:.2f}%  HYG {at_ann['HYG']:.2f}%  MUB {at_ann['MUB']:.2f}%\")\n"
            "print(f\"after-tax exSharpe: HYD {at_sh['HYD']:+.2f}  HYG {at_sh['HYG']:+.2f}  MUB {at_sh['MUB']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: pre-tax, taxable junk (HYG {R['income']['HYG']:.1f}%) out-yields "
            f"HY-muni ({R['income']['HYD']:.1f}%). After the tax break flips it, HYD's **TEY is "
            f"{R['tey']:.1f}%** and its **after-tax return ({R['aftertax_ann']['HYD']:.1f}%) beats "
            f"HYG's ({R['aftertax_ann']['HYG']:.1f}%)**. But the after-tax *risk-adjusted* race is a "
            f"near-tie (HYD {R['aftertax_sharpe']['HYD']:.2f} vs HYG {R['aftertax_sharpe']['HYG']:.2f} "
            f"vs MUB {R['aftertax_sharpe']['MUB']:.2f}) — the wrapper makes HY-muni *competitive*, not "
            "*dominant*."
        ),
        md(
            "### 4d · H₃ — risk & costs\n\n"
            "The drawdown bill (daily total-return) and the one-switch friction bill."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dd = {tk: st.max_drawdown(PX[tk].loc['2009-02-01':])['depth_pct'] for tk in ('HYD','MUB','HYG')}\n"
            "else:\n"
            "    dd = R['dd']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))\n"
            "a1.bar(['HYD','MUB','HYG'], [dd['HYD'], dd['MUB'], dd['HYG']], color=[RED, GREY, AMBER], width=.55)\n"
            "for i, k in enumerate(['HYD','MUB','HYG']): a1.annotate(f'{dd[k]:.1f}%', (i, dd[k]), ha='center', va='top')\n"
            "a1.set_ylabel('max drawdown (%)'); a1.set_title('The risk bill: HY-muni ~22pp deeper than safe muni')\n"
            "cost_x = [f'{c[0]:.0f} bps' for c in R['costs']]\n"
            "a2.bar(cost_x, [c[2] for c in R['costs']], color=GREEN, width=.5, label='net spread (bps/yr)')\n"
            "a2.bar(cost_x, [c[1] for c in R['costs']], color=RED, width=.5, label='cost drag (bps/yr)')\n"
            "for i, c in enumerate(R['costs']): a2.annotate(f'+{c[2]}', (i, c[2]), ha='center', va='bottom')\n"
            "a2.set_ylabel('bps per year'); a2.set_title('The friction bill: invisible (one switch)'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('max DD:', {k: round(v,1) for k,v in dd.items()})\n"
            "for c in R['costs']: print(f'{c[0]:>5.1f} bps one-way: drag {c[1]:.1f} bps/yr -> net {c[2]:+d} bps/yr (gross {R[\"gross_bps\"]:+d})')"
        ),
        md(
            f"> 💡 In plain words: the trade is a **single substitution**, so even 30 bps one-way costs "
            f"only {R['costs'][2][1]:.1f} bps/yr against a **+{R['costs'][2][2]} bps/yr** net spread — "
            "friction does **not** kill it, which is why this is Fragile (thin/crisis-fragile), not "
            f"Mirage (cost-erased). The real bill is the **{R['dd']['HYD']:.0f}% vs {R['dd']['MUB']:.0f}%** "
            "drawdown and the crisis inversion: you warehouse illiquid muni-junk exactly when everyone "
            "wants out."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "A deterministic world HYD = premium + 1.15·MUB + ε with a planted knob. The pipeline must "
            "NOT manufacture a premium from the null and MUST recover a planted +3%/yr."
        ),
        code(
            "res = []\n"
            "for planted in (0.0, 0.03):\n"
            "    w = data.synthetic_world(premium_annual=planted, seed=887)\n"
            "    d = st.synthetic_detect(w)\n"
            "    res.append((planted, d['mean_bps'], d['tstat'], d['ci_low_bps'], d['ci_high_bps']))\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "x = np.arange(2)\n"
            "ax.bar(x, [r[2] for r in res], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(res): ax.annotate(f't={r[2]:.2f}\\nCI[{r[3]:.0f},{r[4]:.0f}]', (i, r[2]), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['planted 0%/yr (null)', 'planted +3%/yr'])\n"
            "ax.set_ylabel('HAC t of the premium'); ax.set_title('Null stays quiet (CI to zero); planted premium lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for p, m_, t, lo, hi in res: print(f'planted {p*100:+.1f}%/yr: mean {m_:+.2f} bps/mo  HAC t {t:+.2f}  CI [{lo:+.1f}, {hi:+.1f}]')"
        ),
        md(
            f"> 💡 In plain words: with **zero** planted premium the machinery reads t = "
            f"{R['syn'][0][2]:.2f} with a CI that straddles zero (quiet); with **+3%/yr** planted it "
            f"reads t ≈ {R['syn'][1][2]:.1f} with a CI clear of zero (loud). The HAC/bootstrap pipeline "
            "is faithful. *(A machinery proof only — never cited in support of the real-tape stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — HYD − MUB **+{R['spread_bps']:.2f} bps/mo** (HAC "
            f"**t = {R['spread_t']:.2f}**, n = {R['n_months']}); bootstrap 95% CI "
            f"**[{R['boot_lo']:.2f}, {R['boot_hi']:.2f}]** touches zero; neither sub-era clears t = 2; "
            "the 2022 spread is **−78 bps/mo (t = −4.35)**. Positive, thin, crisis-fragile — Weak, not "
            "Real (and not None: the sign is right and the Sharpe edge is sign-consistent, +0.13/+0.15).\n"
            f"- **Tradability `FRAGILE`** — one switch, drag ≤ {R['costs'][2][1]:.1f} bps/yr vs "
            f"**+{R['costs'][2][2]} bps/yr net** (costs do **not** erase it → not a Mirage). Fragile: "
            f"the Signal is thin, the payoff crisis-fragile (DD {R['dd']['HYD']:.1f}% vs "
            f"{R['dd']['MUB']:.1f}%), and the mechanical tax pickup (TEY {R['tey']:.1f}% vs "
            f"{R['income']['HYG']:.1f}%; after-tax {R['aftertax_ann']['HYD']:.1f}% > "
            f"{R['aftertax_ann']['HYG']:.1f}%) only helps top-bracket taxable accounts and ties the "
            "after-tax Sharpe race.\n"
            "- **Credit + tax wrapper? `HALF TRUE`** — the wrapper is real and mechanical; the credit "
            "premium over IG munis does not certify."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **It's a liquidity premium, not a default premium.** Munis rarely default; Schwert "
            "(2017) shows the muni spread is largely liquidity. That is *why* it inverts in 2020/2022 — "
            "a premium paid in calm and clawed back in a scramble is structurally Fragile, whatever its "
            "calm-period average.\n"
            "- **The tax leg is contract, not opinion.** As long as muni coupons are federally exempt "
            "and you are top-bracket in a taxable account, TEY 9.2% vs 6.0% holds every month — but it "
            "is a *wrapper* advantage, not a risk-adjusted edge, so it earns credit without a green "
            "stamp.\n"
            "- **Siblings:** [Study 610 — Fallen-Angels](../../610-fallen-angels-premium/) is the "
            "taxable cousin (a within-junk selection premium that **does** clear t ≥ 2); [Study 576 — "
            "Muni-Treasury-Ratio](../../576-muni-treasury-ratio/) (a valuation-timing ratio) and "
            "[Study 616 — Muni-CEF-Tax-Loss](../../616-muni-cef-tax-loss/) (a seasonal CEF effect) test "
            "other muni folklore; [Study 115 — Credit-Spreads](../../115-credit-spreads/) uses credit "
            "as an equity-timing signal.\n\n"
            "*The reproducible core is offline and deterministic; every number above is printed by "
            "[`examples/verify.py`](../examples/verify.py) and frozen in "
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
