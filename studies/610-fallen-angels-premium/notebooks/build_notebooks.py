"""Generate the two narrative notebooks for Study 610 (Fallen-Angels-Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF panel
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic control runs anywhere with no network.
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
# net-of-fee ETF tape, as-of 2026-06-30, fingerprint 2009967185e2).
R = dict(
    asof="2026-06-30", fingerprint="2009967185e2",
    start="2012-05", end="2026-06", n_months=170, years=14.2,
    spread_bps=18.30, spread_t=2.44, spread_pp=2.18,
    ann=dict(ANGL=6.78, HYG=4.60, JNK=4.51),
    sharpe=dict(ANGL=0.63, HYG=0.46, JNK=0.43),
    # alpha specs: (label, alpha bps/mo, HAC t, beta_HY, beta_IEF or None)
    specs=[("ANGL ~ HYG",             15.03, 2.36, 1.122, None),
           ("ANGL ~ HYG + IEF",       15.10, 2.38, 1.119, 0.011),
           ("ANGL ~ HYG + IEF + LQD", 15.83, 2.65, 0.749, -0.445),
           ("ANGL ~ JNK",             16.70, 2.86, 1.078, None),
           ("ANGL ~ JNK + IEF",       16.87, 2.92, 1.071, 0.033)],
    t_ief=0.2,                      # HAC t of the IEF loading in the 2-factor honesty check
    faln=dict(n=120, start="2016-07", spread=14.16, spread_t=1.64,
              alpha=9.46, alpha_t=1.43, dur_alpha=7.64, dur_t=1.22),
    halves=[("2012-05 - 2019-06", 24.77, 2.28, 86),
            ("2019-07 - 2026-06", 11.68, 1.13, 84)],
    decay_diff=13.09, decay_t=0.83,
    dd=dict(ANGL=-29.3, HYG=-22.0),
    dd_dates=dict(ANGL="2020-02-14 - 2020-03-19", HYG="2020-02-13 - 2020-03-23"),
    # costs: (one-way bps, drag bps/yr, net spread bps/yr)
    costs=[(2.0, 0.3, 219), (5.0, 0.7, 219), (10.0, 1.4, 218)],
    # synthetic: (planted bps/mo, a1, t1, a2, t2)
    syn=[(0.0, 5.87, 1.24, 2.61, 0.67), (20.0, 25.87, 5.47, 22.61, 5.78)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Investable](https://img.shields.io/badge/Tradability-Investable-2ea44f?style=flat-square)\n"
    "![Just duration + beta?: Busted](https://img.shields.io/badge/Just_duration_%2B_beta%3F-Busted-8b949e?style=flat-square)\n\n"
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

from fallen_angels_premium import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_prices()                       # sliced to the frozen as-of 2026-06-30
    M = st.monthly_returns(PX)
    MC = st.align_common(M, ["ANGL", "HYG", "JNK", "IEF", "LQD", "BIL"])   # ANGL era
else:
    PX = M = MC = None
print("real ETF cache present:", HAVE_REAL,
      "| ANGL-era months:", (0 if MC is None else len(MC)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Fallen angels: when a bond gets kicked out of the club, buy it 😇\n"
            "### The forced-seller premium in high-yield bonds — tested live, in plain English\n\n"
            + BADGES +
            "A *fallen angel* is a bond that was born respectable — **investment-grade** — and then "
            "got downgraded to junk. Here's the strange part: a huge chunk of the bond market is "
            "**not allowed** to hold junk. Insurance companies, pension funds and investment-grade "
            "index funds *must* sell the moment the rating drops — whatever the price. The claim: "
            "that wave of **forced selling** pushes the price below fair value, and whoever buys the "
            "dumped bonds earns extra return as they recover.\n\n"
            "For once, we don't need to reconstruct history from dusty index tables: since 2012 there "
            "has been a real, fee-charging ETF that holds **only fallen angels** (ANGL) and a real "
            "benchmark that holds ordinary junk bonds (HYG). The market has been running this "
            "experiment for **14 years, live, with real money**. We just read the tape.\n\n"
            "> 📓 **Plain-language layer.** Want the regressions and the robustness checks? See "
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
            "| Do the kicked-out bonds beat ordinary junk bonds? | **Yes — really.** The fallen-angel "
            f"ETF has beaten the broad junk ETF by about **+{R['spread_pp']:.1f}% per year for "
            f"{R['years']:.0f} years**, and the gap is statistically solid, not luck. |\n"
            "| Is it just a riskier portfolio in disguise? | **Mostly no.** It does carry a bit more "
            "market sensitivity, but the extra return survives every \"maybe it's just more "
            "duration / more risk\" control we throw at it. |\n"
            "| Can you actually collect it? | **Yes.** The whole thing is packaged in one liquid ETF. "
            "You buy ANGL instead of HYG — one trade, costs measured in *fractions* of a basis point "
            "per year. |\n"
            "| What's the catch? | **The crash.** In March 2020 the fallen-angel fund fell "
            f"**{R['dd']['ANGL']:.0f}%** while ordinary junk fell **{R['dd']['HYG']:.0f}%**. You are "
            "paid for wearing the deeper drawdown, and the premium arrives in bursts after downgrade "
            "waves — not smoothly. |"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a company's bonds get downgraded from investment-grade to junk, armies of "
            "rule-bound investors are **forced to sell at any price**. The bonds land in the junk "
            "market oversold — and then outperform as they recover (some even get re-upgraded: "
            "'rising stars').\"*\n\n"
            "This is not folklore — it's one of the best-documented effects in bond-market research "
            "(Ben Dor & Xu 2011; Ellul, Jotikasthira & Lundblad 2011 showed insurance companies "
            "literally *fire-sale* downgraded bonds because capital rules make them). Index studies "
            "going back to 1997 all show fallen angels beating ordinary junk. But index studies pay "
            "no fees and trade at fantasy prices — so our desk grades **only the live tape**: the "
            "actual ETF against the actual benchmark, both net of their real fees."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this premium is real *and* collectible, it's one of the rare cases where a retail "
            "investor can pick up an institutional inefficiency with **one trade**: the forced "
            "sellers are structurally locked in (the rules that force them to dump aren't going "
            "away), and the recovery accrues to whoever warehouses the bonds. If it's *not* real — "
            "if it's just a riskier portfolio dressed up — then ANGL is marketing, and you're better "
            "off in the cheaper, broader fund. Billions of dollars sit on each side of that question."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"Simple and brutal: take every complete month since the fallen-angel ETF listed "
            f"(**{R['start']} → {R['end']}, {R['n_months']} months**) and compare **ANGL** (only "
            "fallen angels) with **HYG** (ordinary junk), total-return and net of each fund's real "
            "fee. Then:\n\n"
            "1. **Measure the gap** — the monthly return difference, with statistics that don't get "
            "fooled by streaky months.\n"
            "2. **Try to kill it** — maybe ANGL just holds *longer-duration* or *more market-"
            "sensitive* bonds? We add those factors and see if the extra return dies.\n"
            "3. **Price the pain** — compare the crashes, and the cost of actually doing the trade."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Growth of $100** in the fallen-angel ETF vs the broad junk ETF, since the day both traded."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = PX[[\"ANGL\", \"HYG\"]].loc[\"2012-04-30\":].dropna()\n"
            "    g = 100.0 * px / px.iloc[0]\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(g.index, g[\"ANGL\"], c=GREEN, lw=1.8, label=\"ANGL - fallen angels only\")\n"
            "    ax.plot(g.index, g[\"HYG\"], c=GREY, lw=1.8, label=\"HYG - ordinary junk bonds\")\n"
            "    ax.set_ylabel(\"growth of $100 (total return, net of fees)\")\n"
            "    ax.set_title(\"14 years live: the kicked-out bonds beat ordinary junk\")\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final: ANGL ${g[\"ANGL\"].iloc[-1]:.0f}  vs  HYG ${g[\"HYG\"].iloc[-1]:.0f}')\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', R['ann'])"
        ),
        md(
            f"The fallen-angel fund compounds at **{R['ann']['ANGL']:.2f}%/yr** vs "
            f"**{R['ann']['HYG']:.2f}%/yr** for ordinary junk — a gap of **+{R['spread_pp']:.1f} "
            f"percentage points every year**, after fees, for {R['years']:.0f} years. The quants "
            f"notebook confirms the gap is statistically real (HAC *t* = {R['spread_t']:.2f}, above "
            "the desk's *t* ≥ 2 bar) — not one lucky stretch."
        ),
        md(
            "**Is it just a riskier portfolio in disguise?** The two classic outs: maybe fallen "
            "angels are just *longer-duration* bonds (they were born investment-grade, so they're "
            "longer), or just *more market-sensitive*. We regress the extra return on those factors — "
            "if the \"premium\" is a costume, it dies; if it's real, it survives."
        ),
        code(
            "labels = [s[0] for s in R['specs']]\n"
            "tvals  = [s[2] for s in R['specs']]\n"
            "if HAVE_REAL:\n"
            "    rr = [st.capm_alpha(MC, 'ANGL', 'HYG'),\n"
            "          st.factor_alpha(MC, 'ANGL', ['HYG', 'IEF']),\n"
            "          st.factor_alpha(MC, 'ANGL', ['HYG', 'IEF', 'LQD']),\n"
            "          st.capm_alpha(MC, 'ANGL', 'JNK'),\n"
            "          st.factor_alpha(MC, 'ANGL', ['JNK', 'IEF'])]\n"
            "    tvals = [r['t_alpha'] for r in rr]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(range(len(tvals)), tvals, color=GREEN, width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, label='significance bar (t=2)')\n"
            "for i, v in enumerate(tvals): ax.annotate(f't={v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_xticks(range(len(labels))); ax.set_xticklabels([l.replace(' ~ ', '\\nvs ') for l in labels], fontsize=8)\n"
            "ax.set_ylabel('strength of the leftover extra return (t)')\n"
            "ax.set_title('The extra return survives every \"it is just more risk\" control')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('alpha t across specs:', [round(v, 2) for v in tvals])"
        ),
        md(
            "Every bar clears the significance line. Adding the *duration* factor changes nothing "
            f"(its loading is basically zero, *t* = {R['t_ief']:.1f}); swapping the benchmark (JNK) "
            "makes it *stronger*. About 3 of the 18 monthly basis points **are** just extra market "
            "sensitivity — the honest deduction — but the remaining **~15 bps/month of genuine "
            "alpha** refuses to die. The forced-seller story fits; the costume story doesn't."
        ),
        md(
            "**The catch — the crash you carry.** Nothing in markets is free; here's the bill, "
            "in drawdown form, around COVID (the one big downgrade wave of the sample)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    px = PX[[\"ANGL\", \"HYG\"]].loc[\"2012-04-11\":].dropna()\n"
            "    dd = px / px.cummax() - 1.0\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(dd.index, dd[\"ANGL\"]*100, c=RED, lw=1.2, label=\"ANGL drawdown\")\n"
            "    ax.plot(dd.index, dd[\"HYG\"]*100, c=GREY, lw=1.2, label=\"HYG drawdown\")\n"
            "    ax.set_ylabel(\"drawdown from peak (%)\")\n"
            "    ax.set_title(\"The bill: a deeper crash when downgrades hit (March 2020)\")\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'max DD: ANGL {dd[\"ANGL\"].min()*100:.1f}%  vs  HYG {dd[\"HYG\"].min()*100:.1f}%')\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', R['dd'])"
        ),
        md(
            f"In March 2020 the fallen-angel fund dropped **{R['dd']['ANGL']:.1f}%** peak-to-trough "
            f"vs **{R['dd']['HYG']:.1f}%** for ordinary junk — about **7 points deeper**. That's the "
            "price of admission: when a downgrade wave hits, you're the one catching the falling "
            "bonds. The premium is your pay for that job."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** +{R['spread_pp']:.1f}%/yr over ordinary junk for "
            f"{R['years']:.0f} live years, statistically solid, and it survives the duration and "
            "risk-costume controls.\n"
            "- **Tradability — Investable.** One liquid ETF, one switch, cost drag near zero, fees "
            "already inside the numbers. Rare on this desk.\n"
            "- **\"Just duration + more risk\"? — Busted.** The duration factor loads at zero; the "
            "beta deduction takes ~3 bps of 18; the ~15 bps/month leftover is genuine alpha.\n\n"
            "The honest caveats, out loud: the newer copycat fund (FALN, 2016→) points the same way "
            "but its shorter window can't certify alone; the last seven years alone are positive but "
            "noisier (the premium arrives in bursts after downgrade waves); and you wear a deeper "
            "crash when the wave hits."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why doesn't it get arbitraged away?** The sellers aren't stupid — they're "
            "*constrained*. Capital rules and index mandates force the dump; as long as those rules "
            "exist, someone gets paid to warehouse the bonds. Structural, not informational.\n"
            "- **Why is the premium bursty?** It needs raw material: downgrade waves (2016 energy, "
            "2020 COVID). Quiet years mint few fallen angels, so the edge idles.\n"
            "- **Related desks:** [Study 115 — Credit-Spreads](../../115-credit-spreads/) (do HY "
            "spreads predict *equities*?) and [Study 340 — Bank-Loans](../../340-bank-loans/) (is "
            "floating-rate credit a *safe bond substitute*?) test the other legs of the credit "
            "folklore.\n\n"
            "*Think the premium dies as ETFs crowd the trade? The 2019→2026 half is the place to "
            "look — bring a longer tape and we'll re-grade.*"
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
            "# The Fallen-Angels premium — a quantitative teardown 🔬\n"
            "### HAC excess spread · CAPM-style alpha · duration/quality tilt controls · FALN "
            "confirmation · half-split decay test · drawdowns & costs · synthetic faithful-engine "
            "control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "fallen-angel premium (Ben Dor & Xu 2011; Ellul-Jotikasthira-Lundblad 2011 for the "
            "forced-seller mechanism) is graded here **on the live vehicles only**: ANGL vs HYG/JNK "
            "2012→2026, FALN 2016→ as the independent confirmation, IEF/LQD as tilt controls, BIL "
            "as the tradable risk-free. Index-era studies (1997→) are context, never evidence.\n\n"
            "> ⚠️ **Data note.** yfinance auto-adjusted closes = **total-return, net-of-fee** ETF "
            "tape (ANGL ER 0.35%, HYG 0.49%, FALN 0.25% already inside). As-of **" + R["asof"] + "** "
            "(last complete month), fingerprint `" + R["fingerprint"] + "`. Exactly **one execution "
            "lag**: entry at the 2012-04-30 month-end close, returns accrue from 2012-05. Numbers in "
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
            f"| **Signal** | `REAL` | ANGL − HYG = **+{R['spread_bps']:.2f} bps/mo** (HAC "
            f"**t = {R['spread_t']:.2f}**, n = {R['n_months']}); CAPM-style alpha "
            f"**+{R['specs'][0][1]:.2f} bps/mo** (t = {R['specs'][0][2]:.2f}); alpha t = "
            f"**2.36–2.92** across all five specs. Caveats: FALN alone t = {R['faln']['alpha_t']:.2f}; "
            f"second half alone t = {R['halves'][1][2]:.2f} (decay not certified, Welch "
            f"t = {R['decay_t']:.2f}). |\n"
            f"| **Tradability** | `INVESTABLE` | One switch; drag ≤ {R['costs'][2][1]:.1f} bps/yr vs "
            f"**+{R['costs'][2][2]} bps/yr net**; $3bn AUM, penny spreads, fees inside the tape. "
            f"Price = risk: β = {R['specs'][0][3]:.2f}, DD **{R['dd']['ANGL']:.1f}% vs "
            f"{R['dd']['HYG']:.1f}%**. |\n"
            f"| **Just duration + beta?** | `BUSTED` | β_IEF = **+0.011 (t = {R['t_ief']:.1f})** after "
            f"the HY control — alpha unchanged (+{R['specs'][0][1]:.2f} → +{R['specs'][1][1]:.2f}); "
            f"β_HYG = {R['specs'][0][3]:.2f} explains ~3 of 18 bps/mo, the rest is alpha. |\n\n"
            "> 💡 In plain words: the ejected bonds genuinely out-earn ordinary junk, the extra "
            "return is not a duration or beta costume, and the whole thing is collectible in one "
            "cheap ETF trade — your compensation for warehousing the forced sellers' inventory and "
            "wearing the deeper crash."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "IG mandates (insurance RBC rules, IG-only index funds) **must** sell a bond downgraded "
            "below BBB−. The supply shock is concentrated around index ejection; the natural buyers "
            "(HY funds) are smaller and slower. Prices overshoot; subsequent recovery (plus the "
            "rising-star upgrade option) accrues to the holder. Formally, with monthly excess "
            "returns $r^e$ over the T-bill (BIL):\n\n"
            "$$r^e_{ANGL,t} = \\alpha + \\beta_{HY}\\, r^e_{HYG,t} "
            "+ \\beta_{rate}\\, r^e_{IEF,t} + \\varepsilon_t$$\n\n"
            "- **H₁ (premium exists).** Mean(ANGL − HYG) > 0 with HAC t ≥ 2.\n"
            "- **H₂ (it's alpha, not tilt).** $\\alpha > 0$ with HAC t ≥ 2 **after** the credit-beta "
            "and rate-duration controls.\n"
            "- **H₃ (it's collectible).** The one-switch cost drag is negligible vs the net spread.\n\n"
            "We find **all three supported** — with the FALN short-window and second-half caveats "
            "stated out loud."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The dangerous failure mode of every \"smart\" bond ETF pitch is a **tilt in costume**: "
            "longer duration or higher beta re-labelled as skill. Fallen angels are *specifically* "
            "suspect on both counts — they were issued as IG (longer maturities, lower coupons → "
            "**longer duration** than original-issue junk) and they enter the portfolio at maximum stress "
            "(**higher beta**). So the honesty check is not optional decoration; it *is* the test. "
            "If α dies when IEF enters, the premium is a rate bet. If α dies when β is priced, it's "
            "leverage. Only the surviving residual earns the stamp."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance total-return net-of-fee closes; ANGL-era common sample "
            f"**{R['start']} → {R['end']}** ({R['n_months']} months); as-of {R['asof']}; partial "
            "months dropped; fingerprint `" + R["fingerprint"] + "`.\n"
            "- **H₁.** HAC (Newey-West, Bartlett, rule-of-thumb lags) t on the monthly ANGL − HYG "
            "spread.\n"
            "- **H₂.** Excess-vs-excess OLS on BIL with NW errors: ANGL ~ HYG, then + IEF (duration), "
            "then + LQD (IG credit); benchmark swap to JNK. Five specifications, one bar: t ≥ 2 on α.\n"
            "- **Confirmation.** FALN (independent issuer + index, 2016-07 →): same battery.\n"
            "- **Decay.** Pre-registered midpoint split (2019-06); Welch t on the *difference* "
            "between halves — a decay claim needs the difference, not one quiet half.\n"
            "- **Risk & costs.** Daily max-DD comparison; one-switch drag = 2 legs × one-way bps "
            "amortised over 14.2 yrs; excess Sharpe race.\n"
            "- **Machinery proof.** Synthetic world FA = α + 1.05·HY + 0.25·RATE + ε with a planted-α "
            "knob: null must stay under t = 2, planted +20 bps/mo must light up. Never market "
            "evidence."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · H₁ — the raw spread and its HAC t\n\n"
            "Cumulative relative strength (ANGL/HYG, log scale) plus the monthly spread's HAC test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    diff = (MC['ANGL'] - MC['HYG']).dropna()\n"
            "    h = st.hac_mean(diff)\n"
            "    rel = (1 + MC[['ANGL','HYG']]).cumprod()\n"
            "    ratio = rel['ANGL'] / rel['HYG']\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(ratio.index, ratio, c=GREEN, lw=1.6)\n"
            "    ax.axhline(1.0, c=GREY, ls='--', lw=1)\n"
            "    ax.set_yscale('log'); ax.set_ylabel('ANGL / HYG cumulative ratio (log)')\n"
            "    ax.set_title(f\"Relative strength: +{h['alpha_bps']:.2f} bps/mo, HAC t = {h['t_alpha']:.2f}\")\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f\"ANGL - HYG: {h['alpha_bps']:+.2f} bps/mo  HAC t = {h['t_alpha']:+.2f}  \"\n"
            "          f\"(n={h['n']}, lags={h['lags']})\")\n"
            "    print(f\"annualised: ANGL {st.ann_return(MC['ANGL']):.2f}%  HYG {st.ann_return(MC['HYG']):.2f}%\")\n"
            "    print(f\"excess Sharpe (rf=BIL): ANGL {st.sharpe_excess(MC,'ANGL'):.2f}  \"\n"
            "          f\"HYG {st.sharpe_excess(MC,'HYG'):.2f}  JNK {st.sharpe_excess(MC,'JNK'):.2f}\")\n"
            "else:\n"
            "    print('cache missing - frozen:', R['spread_bps'], R['spread_t'], R['sharpe'])"
        ),
        md(
            f"> 💡 In plain words: **+{R['spread_bps']:.2f} bps/mo** for {R['years']:.1f} years at "
            f"**HAC t = {R['spread_t']:.2f}** — the gap is persistent, not one lucky year (the ratio "
            "chart climbs through most of the sample, with the 2020 downgrade wave as the big "
            f"step). Excess Sharpe race: ANGL **{R['sharpe']['ANGL']:.2f}** vs HYG "
            f"**{R['sharpe']['HYG']:.2f}** — better *risk-adjusted*, not just higher-octane."
        ),
        md(
            "### 4b · H₂ — the alpha gauntlet (five specifications)\n\n"
            "Excess-vs-excess NW regressions. The tilt hypotheses get every chance to kill the alpha: "
            "rate duration (IEF), IG credit (LQD), benchmark swap (JNK)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    specs = [('ANGL ~ HYG',             st.capm_alpha(MC,'ANGL','HYG')),\n"
            "             ('ANGL ~ HYG + IEF',       st.factor_alpha(MC,'ANGL',['HYG','IEF'])),\n"
            "             ('ANGL ~ HYG + IEF + LQD', st.factor_alpha(MC,'ANGL',['HYG','IEF','LQD'])),\n"
            "             ('ANGL ~ JNK',             st.capm_alpha(MC,'ANGL','JNK')),\n"
            "             ('ANGL ~ JNK + IEF',       st.factor_alpha(MC,'ANGL',['JNK','IEF']))]\n"
            "    rows = [(lab, r['alpha_bps'], r['t_alpha']) for lab, r in specs]\n"
            "else:\n"
            "    rows = [(s[0], s[1], s[2]) for s in R['specs']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "x = np.arange(len(rows))\n"
            "a1.bar(x, [r[1] for r in rows], color=GREEN, width=.6)\n"
            "for i, r in enumerate(rows): a1.annotate(f'{r[1]:+.1f}', (i, r[1]), ha='center', va='bottom', fontsize=9)\n"
            "a1.set_xticks(x); a1.set_xticklabels([r[0].replace(' ~ ','\\n~ ') for r in rows], fontsize=7.5)\n"
            "a1.set_ylabel('alpha (bps/month)'); a1.set_title('Alpha level across specs')\n"
            "a2.bar(x, [r[2] for r in rows], color=GREEN, width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, r in enumerate(rows): a2.annotate(f'{r[2]:.2f}', (i, r[2]), ha='center', va='bottom', fontsize=9)\n"
            "a2.set_xticks(x); a2.set_xticklabels([r[0].replace(' ~ ','\\n~ ') for r in rows], fontsize=7.5)\n"
            "a2.set_ylabel('HAC t of alpha'); a2.set_ylim(0, 3.6); a2.set_title('Alpha t across specs'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for lab, a, t in rows: print(f'{lab:26s} alpha {a:+6.2f} bps/mo  HAC t {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the alpha sits at **+15 to +17 bps/mo with t = 2.36–2.92 in all "
            f"five specs**. The duration factor is dead on arrival (β_IEF = "
            f"+{R['specs'][1][4]:.3f}, t = {R['t_ief']:.1f}) — fallen angels *do* run longer duration "
            "in yield terms, but over this sample the HY factor already absorbs it. The honest "
            f"deduction is beta: β_HYG = {R['specs'][0][3]:.2f} converts ~3 bps/mo of the raw spread "
            "into priced risk. What's left is the forced-seller alpha."
        ),
        md(
            "### 4c · Confirmation & decay — FALN and the half-split\n\n"
            "An independent issuer/index (FALN, 2016-07 →) and a pre-registered midpoint split of "
            "the ANGL − HYG spread. A decay claim needs a significant *difference* between halves."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mf = st.align_common(M, ['FALN','HYG','IEF','BIL'])\n"
            "    cf = st.capm_alpha(mf, 'FALN', 'HYG')\n"
            "    diff = (MC['ANGL'] - MC['HYG']).dropna()\n"
            "    h1, h2 = diff.loc[:'2019-06-30'], diff.loc['2019-07-01':]\n"
            "    s1, s2 = st.hac_mean(h1), st.hac_mean(h2)\n"
            "    wt = st.welch_t(h1, h2)\n"
            "    vals = [(f'ANGL half-1\\n2012-2019', s1['alpha_bps'], s1['t_alpha']),\n"
            "            (f'ANGL half-2\\n2019-2026', s2['alpha_bps'], s2['t_alpha']),\n"
            "            ('FALN alpha\\n2016-2026', cf['alpha_bps'], cf['t_alpha'])]\n"
            "    decay = (wt['diff_bps'], wt['t'])\n"
            "else:\n"
            "    vals = [('ANGL half-1\\n2012-2019', R['halves'][0][1], R['halves'][0][2]),\n"
            "            ('ANGL half-2\\n2019-2026', R['halves'][1][1], R['halves'][1][2]),\n"
            "            ('FALN alpha\\n2016-2026', R['faln']['alpha'], R['faln']['alpha_t'])]\n"
            "    decay = (R['decay_diff'], R['decay_t'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [GREEN if v[2] >= 2 else AMBER for v in vals]\n"
            "ax.bar([v[0] for v in vals], [v[1] for v in vals], color=cols, width=.55)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v[1]:+.1f} bps\\nt={v[2]:.2f}', (i, v[1]), ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('premium / alpha (bps per month)')\n"
            "ax.set_title(f'Same sign everywhere; decay NOT certified (Welch t = {decay[1]:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'half-1 {vals[0][1]:+.2f} (t={vals[0][2]:.2f})  half-2 {vals[1][1]:+.2f} (t={vals[1][2]:.2f})')\n"
            "print(f'decay test: diff {decay[0]:+.2f} bps/mo  Welch t = {decay[1]:+.2f}')\n"
            "print(f'FALN alpha {vals[2][1]:+.2f} bps/mo (t={vals[2][2]:.2f})')"
        ),
        md(
            f"> 💡 In plain words: every window points the **same way**, but only the full ANGL tape "
            f"certifies. The second half (+{R['halves'][1][1]:.1f} bps/mo, t = "
            f"{R['halves'][1][2]:.2f}) and FALN (+{R['faln']['alpha']:.1f} bps/mo, t = "
            f"{R['faln']['alpha_t']:.2f}) are positive but under the bar alone — and the half-to-half "
            f"difference is pure noise (Welch t = {R['decay_t']:.2f}). The premium is **episodic** "
            "(it needs downgrade waves: 2016 energy, 2020 COVID), which is exactly what an "
            "un-decayed but bursty structural premium should look like. We say the caveat out loud "
            "rather than certify decay we can't show."
        ),
        md(
            "### 4d · Risk & costs — what you pay, in both currencies\n\n"
            "The drawdown bill (daily, total-return) and the friction bill (one switch, 2 legs)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dd_a = st.max_drawdown(PX['ANGL'].loc['2012-04-11':])\n"
            "    dd_h = st.max_drawdown(PX['HYG'].loc['2012-04-11':])\n"
            "    dd_pair = (dd_a['depth_pct'], dd_h['depth_pct'])\n"
            "else:\n"
            "    dd_pair = (R['dd']['ANGL'], R['dd']['HYG'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))\n"
            "a1.bar(['ANGL', 'HYG'], dd_pair, color=[RED, GREY], width=.5)\n"
            "for i, v in enumerate(dd_pair): a1.annotate(f'{v:.1f}%', (i, v), ha='center', va='top')\n"
            "a1.set_ylabel('max drawdown (%)'); a1.set_title('The risk bill: 7 pp deeper in March 2020')\n"
            "cost_x = [f'{c[0]:.0f} bps' for c in R['costs']]\n"
            "a2.bar(cost_x, [c[2] for c in R['costs']], color=GREEN, width=.5, label='net spread (bps/yr)')\n"
            "a2.bar(cost_x, [c[1] for c in R['costs']], color=RED, width=.5, label='cost drag (bps/yr)')\n"
            "for i, c in enumerate(R['costs']): a2.annotate(f'+{c[2]}', (i, c[2]), ha='center', va='bottom')\n"
            "a2.set_ylabel('bps per year'); a2.set_title('The friction bill: invisible'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'max DD: ANGL {dd_pair[0]:.1f}%  HYG {dd_pair[1]:.1f}%')\n"
            "for c in R['costs']: print(f'{c[0]:>5.1f} bps one-way: drag {c[1]:.1f} bps/yr -> net {c[2]:+d} bps/yr')"
        ),
        md(
            f"> 💡 In plain words: the strategy is a **single substitution** held for years, so even "
            f"10 bps one-way costs {R['costs'][2][1]:.1f} bps/yr against a **+{R['costs'][2][2]} "
            "bps/yr** net spread — friction is irrelevant, which is why this earns `INVESTABLE`. The "
            f"real bill is the **{R['dd']['ANGL']:.0f}% vs {R['dd']['HYG']:.0f}%** drawdown: you are "
            "the warehouse for the forced sellers' inventory precisely when everything is on fire."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "A deterministic world FA = α + 1.05·HY + 0.25·RATE + ε with a planted-α knob. The "
            "pipeline must NOT manufacture alpha from the null and MUST recover a planted +20 bps/mo."
        ),
        code(
            "res = []\n"
            "for planted in (0.0, 20.0):\n"
            "    w = data.synthetic_world(alpha_bps=planted, seed=610)\n"
            "    r1 = st.nw_ols(w['FA'], w[['HY']])\n"
            "    r2 = st.nw_ols(w['FA'], w[['HY','RATE']])\n"
            "    res.append((planted, r1['alpha_bps'], r1['t_alpha'], r2['alpha_bps'], r2['t_alpha']))\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "x = np.arange(2); wdt = 0.35\n"
            "ax.bar(x - wdt/2, [r[2] for r in res], wdt, color=GREY, label='1-factor alpha t')\n"
            "ax.bar(x + wdt/2, [r[4] for r in res], wdt, color=GREEN, label='2-factor alpha t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['planted 0 bps/mo (null)', 'planted +20 bps/mo'])\n"
            "ax.set_ylabel('HAC t of alpha'); ax.set_title('Null stays quiet; planted alpha lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for p, a1_, t1, a2_, t2 in res:\n"
            "    print(f'planted {p:+5.1f}: 1f alpha {a1_:+.2f} (t={t1:+.2f})  2f alpha {a2_:+.2f} (t={t2:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with **zero** planted alpha the machinery reads t = "
            f"{R['syn'][0][2]:.2f}/{R['syn'][0][4]:.2f} (quiet); with **+20 bps/mo** planted it reads "
            f"t ≈ {R['syn'][1][2]:.1f}–{R['syn'][1][4]:.1f} (loud). The HAC-alpha pipeline is "
            "faithful and does not hallucinate. *(A machinery proof only — never cited in support of "
            "the real-tape stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — ANGL − HYG **+{R['spread_bps']:.2f} bps/mo** (HAC "
            f"**t = {R['spread_t']:.2f}**, n = {R['n_months']}); alpha **+15.03 bps/mo at "
            f"t = 2.36**, holding at **t = 2.36–2.92** across HYG/JNK ± IEF ± LQD. Clears **t ≥ 2** "
            "on the tape in every specification. Caveats named: FALN alone t = 1.43 (same sign, "
            "shorter window), second half alone t = 1.13, decay not certified (Welch t = 0.83).\n"
            f"- **Tradability `INVESTABLE`** — one ETF switch; drag ≤ 1.4 bps/yr vs **+218 bps/yr "
            f"net**; $3bn AUM, penny spreads, fees inside the tape. The admission price is risk "
            f"(β = 1.12, DD {R['dd']['ANGL']:.1f}% vs {R['dd']['HYG']:.1f}%), not friction.\n"
            f"- **Just duration + beta? `BUSTED`** — β_IEF = +0.011 (t = 0.2), alpha unchanged when "
            "the rate factor enters; β_HYG = 1.12 converts only ~3 of 18 bps/mo into priced risk. "
            "The forced-seller mechanism, not a tilt costume, is what the tape supports."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The mechanism has an owner.** Ellul et al. (2011) trace the fire sales to insurance "
            "capital rules — a *regulatory* seller that cannot learn its way out of the trade. "
            "That's why 14 years of ETF assets haven't closed it: the supply of forced sellers is "
            "policy, not opinion.\n"
            "- **Watch the raw material.** The premium needs downgrade waves. A long quiet-credit "
            "regime idles it (the 2019→2026 half alone is t = 1.13); a recession mints it. Sizing it "
            "as a permanent overweight means accepting years of nothing.\n"
            "- **The index-vs-live gap.** Index studies since 1997 show bigger numbers than +2.2 "
            "pp/yr — index math pays no fees and assumes frictionless rebalancing into the most "
            "dislocated bonds. The live, fee-paying, capacity-constrained version is the only one "
            "this desk grades — and it still clears.\n"
            "- **Siblings:** [Study 115 — Credit-Spreads](../../115-credit-spreads/) (credit as an "
            "*equity timing* signal — fails) and [Study 340 — Bank-Loans](../../340-bank-loans/) "
            "(duration swapped for credit risk — a costume this study's controls are built to catch).\n\n"
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
