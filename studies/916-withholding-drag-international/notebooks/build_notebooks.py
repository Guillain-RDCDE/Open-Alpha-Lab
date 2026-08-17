"""Generate the two narrative notebooks for Study 916 (Withholding Drag).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the
fast synthetic control, and they are never presented under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. VEA vs an EWJ/EWU/EWG 50/30/20
# blend, total-return and price-only legs, 2007-07-30 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2007-07-30", end="2026-06-30", n_days=4760, fp="ca173c825c7c",
    # 1. the ruler
    ruler=[("VEA", 302.3, 302.2, +0.10, 66), ("IEFA", 291.3, 291.7, -0.44, 29),
           ("EFA", 269.3, 269.2, +0.04, 47), ("VXUS", 296.7, 297.2, -0.42, 57),
           ("EWJ", 147.6, 147.2, +0.38, 50), ("EWG", 232.7, 232.5, +0.16, 42),
           ("EWU", 373.5, 372.9, +0.60, 56)],
    # 2. resolution check
    cal_n=3437, cal_efa=300.8, cal_iefa=291.3, cal_gap=-9.5, cal_t=-1.05,
    cal_lo=-28.8, cal_hi=6.5, cal_expected=26.0, cal_resid=-35.5,
    # 3. headline table
    head=[("VEA", 4760, 302.4, 254.9, -47.5, -1.07, -0.5, -0.01, -69.3, 69.0),
          ("IEFA", 3436, 291.4, 257.8, -33.6, -2.77, 9.4, 0.78, -12.5, 30.6),
          ("EFA", 6244, 269.3, 222.9, -46.4, -4.11, -29.4, -2.61, -49.3, -10.0),
          ("VXUS", 3875, 296.8, 256.0, -40.8, -0.89, 4.2, 0.09, -56.3, 68.4)],
    fee_gap=47.0,
    # 4. calendar years
    n_years=18, n_negative_years=17, worst_year=2013, worst_gap=-114.6,
    flip_year=2025, flip_gap=28.0,
    # 5. eras
    era_e_n=2375, era_e_gap=-59.2, era_e_t=-0.85, era_e_adj=-12.2,
    era_l_n=2384, era_l_gap=-35.8, era_l_t=-0.62, era_l_adj=11.2,
    # 6. weight sweep
    wsweep=[("EAFE-ish 50/30/20", 254.9, -47.5, -0.5, -1.07),
            ("equal 1/3 each", 271.5, -30.9, 16.1, -0.65),
            ("Japan-heavy 70/20/10", 226.5, -75.9, -28.9, -1.82),
            ("UK-heavy 30/50/20", 298.1, -4.3, 42.7, -0.09)],
    wsweep_span=71.6,
    # 6b. expense-ratio sweep (today's fees applied to a 19-year window)
    fsweep=[(35.0, -12.5, -0.28, -81.3, 57.0), (40.0, -7.5, -0.17, -76.3, 62.0),
            (44.0, -3.5, -0.08, -72.3, 66.0), (47.0, -0.5, -0.01, -69.3, 69.0),
            (51.0, 3.5, 0.08, -65.3, 73.0), (55.0, 7.5, 0.17, -61.3, 77.0)],
    # 7. inference
    net_yield=302.0,
    infer=[(0.05, 318, 16), (0.10, 336, 34), (0.12, 344, 41),
           (0.15, 356, 53), (0.20, 378, 76), (0.25, 403, 101)],
    central_w=0.12, central_drag=41,
    # 8. the race
    race=[(0.0, 0.288, 0.251, -0.037, -102, -1.05),
          (5.0, 0.288, 0.250, -0.038, -103, -1.06),
          (25.0, 0.288, 0.249, -0.040, -107, -1.09)],
    ci_vea_lo=-0.091, ci_vea_hi=0.702, ci_blend_lo=-0.132, ci_blend_hi=0.652,
    # 9. synthetic control
    syn_planted_true=25.6, syn_planted_meas=25.4, syn_planted_t=9.17,
    syn_null_meas=0.1, syn_null_lo=-0.1, syn_null_hi=0.4,
)


HEADER = f"""# Study 916 — Withholding Drag 🌍

**How much of an international fund's dividend is eaten before it reaches you?**

A US-domiciled fund holding Japanese, German or French shares has foreign dividend tax
withheld *at source* — before the cash ever touches the fund's NAV. On a ~3% yield a
10–15% effective rate would be 30–50 bp/yr, bigger than the fund's own fee. This study
tries to **measure** that leak from public data rather than assume it.

The tape: two legs per fund — the **total-return** close and the **price-only** close.
Their difference is the distribution the fund actually paid. Headline pair **VEA**
against an **EWJ/EWU/EWG 50/30/20** blend of the same market, {R['start']} → {R['end']}
({R['n_days']:,} sessions), one execution lag on the rebalance, costs one-way × NAV.

*Real numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fp']}`); the live cells run only the offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. First, a ruler that actually works\n\n"
           "A fund's *total-return* price line includes its dividends; its *price-only* "
           "line does not. Subtract one from the other and what is left is the cash the "
           "fund handed you. Does that trick give the right answer? Compare it against "
           "the dividends the funds actually declared.\n\n"
           "> 🔬 **For the quants:** the daily difference `r_TR − r_PX` is zero off "
           "ex-date and `D_t / P_{t−1}` on it; summing over a year is the realised "
           "distribution yield. Both legs are split-adjusted; only the first is "
           "dividend-adjusted."),
        code(
            "R = %r\n"
            "print('fund   differenced   declared cash   residual')\n"
            "for tk, diff, cash, resid, nex in R['ruler']:\n"
            "    print('%%-6s %%8.1f bp %%12.1f bp %%9.2f bp  (%%d ex-dates)'\n"
            "          %% (tk, diff, cash, resid, nex))\n"
            "print('\\nThe ruler is exact to under a basis point on all seven funds.')"
            % ({"ruler": R["ruler"]},)
        ),
        md("## 2. So we can measure the income. Can we measure the *tax*?\n\n"
           "No — and this is the whole story. The tape shows what the fund **paid you**. "
           "It never shows what the Japanese and German companies **declared** before "
           "the taxman took his cut. To see the difference you would need a benchmark "
           "that was *not* taxed — and every US-listed international ETF is taxed the "
           "same way.\n\n"
           "The plan was to use single-country funds (EWJ, EWU, EWG) as a stand-in for "
           "the untaxed market. Here is what that comparison actually produces."),
        code(
            "R = %r\n"
            "print('fund   yield   blend   raw gap (t)      after fees (t)')\n"
            "for tk, n, f, b, gap, t, adj, tadj, lo, hi in R['head']:\n"
            "    print('%%-6s %%6.1f %%7.1f %%8.1f (%%+.2f) %%12.1f (%%+.2f)'\n"
            "          %% (tk, f, b, gap, t, adj, tadj))\n"
            "print('\\nThe gap is NEGATIVE: the broad funds pay out MORE than the blend,')\n"
            "print('the opposite of what a withholding leak would look like.')"
            % ({"head": R["head"]},)
        ),
        md(f"## 3. Why the gap points the wrong way\n\n"
           f"Two dull reasons, neither of them tax:\n\n"
           f"1. **Fees.** The single-country funds charge **50 bp**; VEA charges 3 bp. "
           f"Distributions are paid *after* fees, so the expensive benchmark hands back "
           f"less income by construction — a **{R['fee_gap']:.0f} bp** head start.\n"
           f"2. **Composition.** Japan yields 148 bp, Germany 233 bp, the UK 374 bp. "
           f"Which three you pick, and in what proportion, moves the answer more than "
           f"any tax could.\n\n"
           f"Add the fee difference back and the headline gap is "
           f"**{R['head'][0][6]:+.1f} bp/yr** — a flat zero (*t* = {R['head'][0][7]:+.2f}). "
           f"Change the country weights and it swings by **{R['wsweep_span']:.0f} bp**."),
        code(
            "R = %r\n"
            "print('weights                blend yield   raw gap   after fees')\n"
            "for name, b, gap, adj, t in R['wsweep']:\n"
            "    print('%%-22s %%9.1f %%10.1f %%11.1f' %% (name, b, gap, adj))\n"
            "print('\\nspan of the fee-adjusted estimate: %%.0f bp/yr -- the assumption,'\n"
            "      %% R['wsweep_span'])\n"
            "print('not the tape, is doing the talking.')"
            % ({"wsweep": R["wsweep"], "wsweep_span": R["wsweep_span"]},)
        ),
        md(f"## 4. What we *can* say — an inference, clearly labelled\n\n"
           f"VEA's measured **net** distribution yield is **{R['net_yield']:.0f} bp/yr**. "
           f"If you are willing to *assume* an effective withholding rate, the gross "
           f"dividend and the drag follow by arithmetic. That is all this is — "
           f"arithmetic on an assumption, not a measurement.\n\n"
           f"> 🔬 **For the quants:** gross = net / (1 − w), drag = net · w / (1 − w). "
           f"The tape identifies `net` and nothing else."),
        code(
            "R = %r\n"
            "print('assumed rate   implied gross   implied drag')\n"
            "for w, gross, drag in R['infer']:\n"
            "    star = '   <- central assumption' if abs(w - R['central_w']) < 1e-9 else ''\n"
            "    print('   %%3.0f%%%%          %%5.0f bp       %%5.0f bp/yr%%s' %% (w*100, gross, drag, star))\n"
            "print('\\nHonest range: 16 to 101 bp/yr. A 6x spread, set entirely by a')\n"
            "print('number the tape cannot see.')"
            % ({"infer": R["infer"], "central_w": R["central_w"]},)
        ),
        md(f"## 5. And could you do anything about it anyway?\n\n"
           f"No. Suppose the drag really is {R['central_drag']} bp/yr. Swapping the cheap "
           f"broad fund for the single-country blend does not avoid it — those funds are "
           f"US-domiciled too, taxed identically — and it costs about **1 percentage "
           f"point a year** in fees and composition (excess-of-cash return difference "
           f"{R['race'][1][4]:+.0f} bp/yr, HAC *t* = {R['race'][1][5]:+.2f}, essentially "
           f"unchanged from 0 to 25 bps of cost).\n\n"
           f"The one lever that *is* real lives on your tax return, not in the market: a "
           f"US **taxable** account can usually claim the foreign tax credit; an IRA or "
           f"401(k) cannot. Which is an argument about *where* you hold the fund, not "
           f"*which* fund you hold."),
        md("## 6. Is the detector broken, or is the answer really null?\n\n"
           "Fair question. We build a fake world where the broad fund genuinely does "
           "lose an extra slice of its dividend to tax, and check the estimator finds "
           "it — then a world where nobody is taxed differently, and check it stays "
           "quiet. This cell runs live, offline."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from withholding import data, strategy as st\n"
            "tr, px, truth = data.synthetic_panel(signal_strength=1.0, seed=916)\n"
            "planted = st.synthetic_detect(tr, px, truth, n_boot=400)\n"
            "tr0, px0, truth0 = data.synthetic_panel(signal_strength=0.0, seed=916)\n"
            "null = st.synthetic_detect(tr0, px0, truth0, n_boot=400)\n"
            "print('planted a %.1f bp leak -> estimator reads %+.1f bp (t %+.2f)'\n"
            "      % (planted['true_gap_bp'], planted['measured_gap_bp'], planted['t_hac']))\n"
            "print('planted NO leak       -> estimator reads %+.1f bp, CI [%+.1f, %+.1f]'\n"
            "      % (null['measured_gap_bp'], null['ci_low_bp'], null['ci_high_bp']))\n"
            "print('\\nThe detector works. The real-tape null is about the benchmark,')\n"
            "print('not about the machinery.')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The question is not answerable from the tape. The "
           f"measurement is exact (the ruler reproduces declared cash to under a basis "
           f"point on all seven funds), but the *gross* dividend never appears in any "
           f"price series, and every candidate benchmark is a US fund taxed the same "
           f"way. The headline gap is **{R['head'][0][4]:+.1f} bp/yr with the wrong "
           f"sign** (HAC *t* = {R['head'][0][5]:+.2f}), collapses to "
           f"**{R['head'][0][6]:+.1f} bp** once fees are added back, is insignificant in "
           f"both eras and flips sign between them, and swings "
           f"{R['wsweep_span']:.0f} bp across plausible country weights.\n"
           f"- **Tradability — Mirage.** Nothing to bank. The 'gross' benchmark is taxed "
           f"identically and costs ~{abs(R['race'][1][4]):.0f} bp/yr more to own. The "
           f"only genuine lever is the foreign tax credit — a tax-return mechanic, "
           f"off-tape.\n"
           f"- **What survives.** The ruler. Total return minus price return measures any "
           f"fund's realised income yield to within a basis point, and the desk keeps it."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 916 — Withholding Drag — the teardown\n\n"
           "The measurement identity and its validation, the resolution floor, the "
           "income-yield gap with HAC *t* and block-bootstrap CIs, the fee decomposition, "
           "the era cut, the weight sweep, the labelled withholding inference, the "
           "excess-of-cash total-return race with a cost sweep, and the live synthetic "
           "control. Every real number is frozen from `docs/results.md` "
           "(fingerprint `%s`, as-of 2026-06-30)." % R["fp"]),
        code("R = %r" % (R,)),
        md("## The identity\n\n"
           "For split-adjusted legs `TR` (dividend-adjusted) and `PX` (not),\n\n"
           "    d_t = TR_t/TR_{t-1} - PX_t/PX_{t-1}\n\n"
           "is 0 off ex-date and `D_t / PX_{t-1}` on it. Summed over 252 sessions it is "
           "the realised **net** distribution yield: net of foreign withholding *and* of "
           "the fund's expenses, because both come out before the cash is declared.\n\n"
           "> 💡 **In plain words:** the total-return line includes the dividends, the "
           "price line does not; the gap between them is the cash you were paid.\n\n"
           "Validation against the funds' declared per-share cash distributions:"),
        code(
            "print(f\"{'fund':6s}{'differenced':>13s}{'declared':>12s}{'residual':>11s}{'ex-dates':>10s}\")\n"
            "for tk, diff, cash, resid, nex in R['ruler']:\n"
            "    print(f'{tk:6s}{diff:11.1f}bp{cash:10.1f}bp{resid:+10.2f}bp{nex:10d}')\n"
            "print('\\nmax |residual| = %.2f bp -> no systematic measurement bias'\n"
            "      % max(abs(r[3]) for r in R['ruler']))"
        ),
        md(f"## The resolution floor — EFA vs IEFA\n\n"
           f"Same issuer, near-identical market, a **known** {R['cal_expected']:.0f} bp fee "
           f"gap (EFA 33 bp, IEFA 7 bp). If the estimator can resolve tens of basis points "
           f"of *fund-level* income difference, it should recover that. It does not — "
           f"IEFA's index includes small caps and EFA's does not, and composition swamps "
           f"the fee signal.\n\n"
           f"> 💡 **In plain words:** even for two near-twins, the honest error bar is "
           f"several tens of basis points — as large as the effect we are hunting."),
        code(
            "print(f\"EFA {R['cal_efa']:.1f} bp/yr vs IEFA {R['cal_iefa']:.1f} bp/yr  (n={R['cal_n']:,})\")\n"
            "print(f\"measured gap {R['cal_gap']:+.1f} bp   HAC t {R['cal_t']:+.2f}   \"\n"
            "      f\"95% CI [{R['cal_lo']:+.1f}, {R['cal_hi']:+.1f}]\")\n"
            "print(f\"fee gap alone predicts {R['cal_expected']:+.1f} bp -> residual {R['cal_resid']:+.1f} bp\")\n"
            "print('\\n-> the +26 bp truth sits OUTSIDE the CI. Resolution floor ~ tens of bp.')"
        ),
        md("## The headline gap — broad fund minus single-country blend\n\n"
           "Blend = EWJ/EWU/EWG 50/30/20, monthly rebalance, weights known at month-end "
           "*t* and traded at *t+1* (the study's single execution lag). Raw gap first, "
           "then with the 47 bp fee difference added back — a constant shift, so the CI "
           "width is unchanged and the HAC *t* is simply re-centred."),
        code(
            "hdr = f\"{'fund':6s}{'n':>7s}{'fund':>8s}{'blend':>8s}{'gap':>9s}{'t':>7s}{'fee-adj':>10s}{'t':>7s}{'CI(fee-adj)':>20s}\"\n"
            "print(hdr)\n"
            "for tk, n, f, b, gap, t, adj, tadj, lo, hi in R['head']:\n"
            "    print(f'{tk:6s}{n:7,d}{f:8.1f}{b:8.1f}{gap:9.1f}{t:+7.2f}{adj:10.1f}{tadj:+7.2f}'\n"
            "          f'   [{lo:+.1f}, {hi:+.1f}]')\n"
            "print('\\nEvery raw gap is NEGATIVE -- the broad funds distribute MORE than the')\n"
            "print('blend, the wrong sign for a withholding leak. The only |t|>2 survivor')\n"
            "print('after fees (EFA, t=-2.61) still points the wrong way and starts in 2001.')"
        ),
        md(f"## Era cut and weight sweep\n\n"
           f"Split at 2017-01-01. The raw gap is insignificant in both halves and the "
           f"fee-adjusted gap **changes sign** ({R['era_e_adj']:+.1f} → "
           f"{R['era_l_adj']:+.1f} bp). The weight sweep is the more damning of the two: "
           f"the composition assumption moves the estimate by "
           f"{R['wsweep_span']:.0f} bp/yr, more than any plausible tax effect.\n\n"
           f"> 💡 **In plain words:** you get whatever answer your choice of benchmark "
           f"countries hands you."),
        code(
            "print(f\"2007-2016 (n={R['era_e_n']:,}): gap {R['era_e_gap']:+.1f} (t {R['era_e_t']:+.2f})  fee-adj {R['era_e_adj']:+.1f}\")\n"
            "print(f\"2017-2026 (n={R['era_l_n']:,}): gap {R['era_l_gap']:+.1f} (t {R['era_l_t']:+.2f})  fee-adj {R['era_l_adj']:+.1f}  <- sign flip\")\n"
            "print()\n"
            "for name, b, gap, adj, t in R['wsweep']:\n"
            "    print(f'{name:24s} blend {b:6.1f}  gap {gap:+7.1f}  fee-adj {adj:+7.1f}  (t {t:+.2f})')\n"
            "print(f'\\nfee-adjusted span across weight sets: {R[\"wsweep_span\"]:.0f} bp/yr')"
        ),
        md("## The third assumption — expense ratios\n\n"
           "The fee add-back uses **today's** (2026) fact-sheet fees on a 19-year window, "
           "and ETF fees fell over it: VEA was well above 10 bp at its 2007 launch against "
           "3 bp now, while the iShares single-country funds barely moved. The "
           "time-averaged fee gap is therefore *smaller* than 47 bp — so the headline "
           "fee-adjusted gap is an **upper bound**, i.e. it already flatters the "
           "withholding hypothesis (which needs a *positive* gap).\n\n"
           "> 💡 **In plain words:** we gave the theory the most generous fee assumption "
           "available and it still did not clear zero."),
        code(
            "print(f\"{'fee gap':>8s}{'fee-adj gap':>13s}{'HAC t':>8s}{'95% CI':>22s}\")\n"
            "for f, adj, t, lo, hi in R['fsweep']:\n"
            "    mark = '  <- headline' if abs(f - 47.0) < 1e-9 else ''\n"
            "    print(f'{f:7.1f}b{adj:+13.1f}{t:+8.2f}   [{lo:+.1f}, {hi:+.1f}]{mark}')\n"
            "span = R['fsweep'][-1][1] - R['fsweep'][0][1]\n"
            "print(f'\\nfee-adjusted gap spans {span:.0f} bp across the fee grid; max |t| = '\n"
            "      f\"{max(abs(r[2]) for r in R['fsweep']):.2f}. No fee assumption rescues \"\n"
            "      'the sign or the significance.')"
        ),
        md(f"## INFERENCE — not a measurement\n\n"
           f"`gross = net / (1 − w)`, `drag = net · w / (1 − w)`. The tape identifies "
           f"`net` = {R['net_yield']:.0f} bp/yr and nothing else, so `w` is an "
           f"**assumption** (Japan 15% and Germany 15% by treaty, the UK 0% because it "
           f"levies no dividend withholding tax at all) and the answer is a sweep, never "
           f"a point.\n\n"
           f"> 💡 **In plain words:** we can tell you exactly what landed in your account "
           f"and only guess what was skimmed on the way."),
        code(
            "print(f\"measured net distribution yield: {R['net_yield']:.0f} bp/yr (VEA)\")\n"
            "for w, gross, drag in R['infer']:\n"
            "    mark = '  <- central ASSUMPTION' if abs(w - R['central_w']) < 1e-9 else ''\n"
            "    print(f'  w={w:5.0%}  implied gross {gross:5.0f} bp  implied drag {drag:5.0f} bp/yr{mark}')\n"
            "lo = R['infer'][0][2]; hi = R['infer'][-1][2]\n"
            "print(f'\\nhonest range {lo}-{hi} bp/yr ({hi/lo:.1f}x), driven purely by w.')\n"
            "print('A US taxable holder recovers much of this via the foreign tax credit;')\n"
            "print('an IRA/401(k) holder does not. Off-tape, not modelled.')"
        ),
        md("## Is it bankable? Excess-of-cash race, VEA vs the blend\n\n"
           "Both legs excess of BIL's own total return. Costs are charged one-way × NAV "
           "on the blend's monthly rebalance turnover only; VEA is buy-and-hold. No short "
           "leg anywhere, so no borrow applies."),
        code(
            "print(f\"{'cost':>6s}{'VEA exSh':>11s}{'blend exSh':>13s}{'adv':>9s}{'ret adv':>11s}{'HAC t':>8s}\")\n"
            "for c, sf, sb, adv, ret, t in R['race']:\n"
            "    print(f'{c:5.1f}b{sf:+11.3f}{sb:+13.3f}{adv:+9.3f}{ret:+9.0f}bp{t:+8.2f}')\n"
            "print(f\"\\nbootstrap excess-Sharpe CI: VEA [{R['ci_vea_lo']:+.3f}, {R['ci_vea_hi']:+.3f}]  \"\n"
            "      f\"blend [{R['ci_blend_lo']:+.3f}, {R['ci_blend_hi']:+.3f}] -> heavily overlapping\")\n"
            "print('The \"gross-yield\" wrapper costs ~1 pp/yr and buys back no measurable tax.')"
        ),
        md("## Live synthetic control — the machinery is unbiased\n\n"
           "A four-fund panel on one market factor. The broad fund loses an extra 8 pp of "
           "its dividend to withholding at `signal_strength=1` (a planted 25.6 bp/yr "
           "leak) and exactly the same as the benchmark at `signal_strength=0`. The "
           "estimator must recover the first and stay silent on the second."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from withholding import data, strategy as st\n"
            "pl = st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, seed=916), n_boot=400)\n"
            "print('planted %.1f bp -> measured %+.1f bp (t %+.2f), CI [%+.1f, %+.1f]'\n"
            "      % (pl['true_gap_bp'], pl['measured_gap_bp'], pl['t_hac'], pl['ci_low_bp'], pl['ci_high_bp']))\n"
            "nulls = np.array([st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=916+s),\n"
            "                                     n_boot=200)['measured_gap_bp'] for s in range(5)])\n"
            "print('null x5: mean %+.2f bp (sd %.2f), |gap|>=10 bp in %d/5'\n"
            "      % (nulls.mean(), nulls.std(ddof=1), int((np.abs(nulls) >= 10).sum())))\n"
            "half = st.synthetic_detect(*data.synthetic_panel(signal_strength=0.5, seed=916), n_boot=200)\n"
            "print('dose response: half the planted leak -> %+.1f bp (half of %+.1f)'\n"
            "      % (half['measured_gap_bp'], pl['measured_gap_bp']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The estimand is **unidentified on the tape**. "
           f"Measurement is exact (max residual "
           f"{max(abs(r[3]) for r in R['ruler']):.2f} bp against declared cash), but the "
           f"gross dividend is never published in a price series and every US-listed "
           f"benchmark suffers the same treaty withholding, so the constructed gap is "
           f"fees plus composition. Headline: **{R['head'][0][4]:+.1f} bp/yr** raw "
           f"(HAC *t* = {R['head'][0][5]:+.2f}, wrong sign), **{R['head'][0][6]:+.1f} bp** "
           f"fee-adjusted (*t* = {R['head'][0][7]:+.2f}, CI "
           f"[{R['head'][0][8]:+.1f}, {R['head'][0][9]:+.1f}]); insignificant in both "
           f"eras with a sign flip between them; {R['wsweep_span']:.0f} bp of swing across "
           f"country weights; and a resolution floor from the EFA/IEFA twin test that "
           f"already fails to recover a *known* {R['cal_expected']:.0f} bp fee gap. No "
           f"|*t*| ≥ 2 in the right direction. The synthetic control recovers a planted "
           f"{R['syn_planted_true']:.1f} bp leak to "
           f"{R['syn_planted_meas']:+.1f} bp and is silent on the null "
           f"({R['syn_null_meas']:+.1f} bp), so the null is the tape's, not the harness's.\n"
           f"- **Tradability — Mirage.** Even granting the inferred "
           f"{R['central_drag']} bp/yr central drag, no US-listed wrapper avoids it; the "
           f"single-country benchmark costs {abs(R['race'][1][4]):.0f} bp/yr more "
           f"excess-of-cash (HAC *t* = {R['race'][1][5]:+.2f}) and is unaffected by cost "
           f"assumptions. The only real lever is the foreign tax credit, which is a "
           f"tax-return mechanic and off-tape.\n"
           f"- **Survivorship & proxies.** The seven funds are today's largest survivors "
           f"(a mild ex-post tilt, named on the Signal axis). Expense ratios (a named "
           f"anachronism — today's fees on a 19-year window, swept "
           f"{R['fsweep'][0][0]:.0f}–{R['fsweep'][-1][0]:.0f} bp), blend weights and the "
           f"withholding rate are labelled assumptions and each is swept."),
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())]:
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
