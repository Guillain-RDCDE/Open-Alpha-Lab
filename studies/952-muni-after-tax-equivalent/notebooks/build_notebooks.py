"""Generate the two narrative notebooks for Study 952 (After-Tax Equivalent).

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


# --------------------------------------------------------------------------- #
# Frozen real-tape headline — the single source of truth, mirroring docs/results.md.
# Monthly, 2004-02 -> 2026-06, total-return AND price-only tapes, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2004-02", end="2026-06", n_months=269, fp="0f379866a58d",
    # reconstructed income legs, annualised %
    inc_mub=2.77, inc_vteb=2.32, inc_sub=1.27, inc_hyd=5.43,
    inc_agg=3.20, inc_lqd=4.07, inc_vcit=3.63, inc_bil=1.35,
    # MUB's income over the shorter windows its comparators impose
    inc_mub_on_vcit=2.66, inc_agg_on_mub=3.00, inc_lqd_on_mub=3.90,
    # break-even effective marginal rates, %
    be_mub_vcit=35.0, be_mub_lqd=29.7, be_vteb_vcit=28.9,
    be_mub_agg=-6.9, be_vteb_agg=-18.7, be_sub_bil=-24.0, be_hyd_lqd=-21.5,
    # ...and how badly identified each of them is (block bootstrap on tau*)
    beci_mub_vcit=(-10.9, 81.5), beci_mub_lqd=(-25.9, 79.3), beci_vteb_vcit=(-27.3, 82.7),
    beci_mub_agg=(-52.1, 37.3), beci_vteb_agg=(-78.1, 38.6), beci_sub_bil=(-115.3, 34.6),
    beci_hyd_lqd=(-94.8, 58.1),
    p_above_top_mub_vcit=0.41, p_below_zero_mub_vcit=0.06,
    price_leg_mub_vcit=-2.53, price_leg_mub_vcit_t=-0.36,
    price_leg_mub_agg=3.64, price_leg_mub_agg_t=0.68,
    # the income-leg-only (tax-equivalent-yield) break-even — the tight one
    ibe_mub_vcit=26.7, ibeci_mub_vcit=(23.4, 29.7),
    ibe_mub_lqd=28.9, ibeci_mub_lqd=(27.1, 30.7),
    ibe_vteb_vcit=34.3, ibeci_vteb_vcit=(31.0, 38.2),
    ibe_mub_agg=7.6, ibeci_mub_agg=(2.7, 12.0),
    # after-tax diff = pre-tax diff + tax term (mean share, variance share)
    dec_mub_agg=(1.73, 0.31, 10.21, 5.72, 11.94, 2.22, 85.5, 0.30),
    dec_vteb_agg=(4.34, 0.65, 9.47, 4.74, 13.81, 2.10, 68.6, 0.37),
    dec_sub_bil=(2.49, 0.69, 4.22, 6.87, 6.71, 2.02, 62.9, 1.87),
    dec_hyd_lqd=(6.77, 0.57, 12.83, 6.01, 19.60, 1.67, 65.5, 0.09),
    # mean-share is None where the pre-tax term has the opposite sign (share is meaningless)
    dec_mub_vcit=(-10.60, -1.50, 12.35, 6.32, 1.75, 0.25, None, 0.32),
    # pre-tax monthly differences, bps
    pre_mub_vcit=-10.60, pre_mub_vcit_t=-1.50,
    pre_mub_lqd=-9.65, pre_mub_lqd_t=-1.04,
    pre_mub_agg=1.73, pre_mub_agg_t=0.31,
    # after-tax race at 40.8%
    n_mub_vcit=199, d_mub_vcit=1.72, t_mub_vcit=0.25,
    ci_mub_vcit_lo=-12.5, ci_mub_vcit_hi=15.6,
    sh_mub_vcit_m=0.47, sh_mub_vcit_t=0.35, ann_mub_vcit=0.21,
    n_mub_lqd=225, d_mub_lqd=3.59, t_mub_lqd=0.39,
    n_mub_agg=225, d_mub_agg=11.91, t_mub_agg=2.22, ann_mub_agg=1.43,
    ci_mub_agg_lo=0.9, ci_mub_agg_hi=22.5,
    sh_mub_agg_m=0.48, sh_mub_agg_t=0.23,
    n_vteb_agg=130, d_vteb_agg=13.77, t_vteb_agg=2.09,
    n_sub_bil=211, d_sub_bil=6.69, t_sub_bil=2.01,
    n_hyd_lqd=208, d_hyd_lqd=19.57, t_hyd_lqd=1.66,
    # bracket ladder, MUB vs VCIT
    ladder_rates=[0.0, 24.0, 27.8, 35.8, 40.8],
    ladder_vcit=[-10.63, -3.37, -2.22, 0.20, 1.72],
    ladder_vcit_t=[-1.50, -0.48, -0.32, 0.03, 0.25],
    ladder_agg=[1.70, 7.71, 8.66, 10.66, 11.91],
    ladder_agg_t=[0.31, 1.42, 1.60, 1.98, 2.22],
    # era cut (split 2017-01)
    era_vcit_e_n=85, era_vcit_e=-2.40, era_vcit_e_t=-0.23,
    era_vcit_l_n=114, era_vcit_l=4.73, era_vcit_l_t=0.51,
    era_agg_e_n=111, era_agg_e=9.96, era_agg_e_t=1.25,
    era_agg_l_n=114, era_agg_l=13.76, era_agg_l_t=1.92,
    # assumption sweeps, MUB vs VCIT at 37% federal
    state_none=1.72, state_ca_out=2.47, state_ca_in=4.53, state_ca_in_t=0.65,
    cg_0=1.72, cg_15=2.10, cg_238=2.32,
    floor_on=1.717, floor_off=1.716,
    # costs and borrow
    cost0=1.75, cost25=1.50,
    borrow0_vcit=1.72, borrow25_vcit=-0.37, borrow100_vcit=-6.62,
    borrow0_agg=11.91, borrow25_agg=9.83, borrow25_agg_t=1.83, borrow100_agg=3.58,
    # the one signal arm
    ov_n=199, ov_switches=3, ov_in=93.0, ov_ann=3.33, ov_hold=3.05,
    ov_edge=2.31, ov_t=0.90,
    # synthetic control
    syn_planted_true=33.33, syn_planted_got=38.97, syn_planted_maxerr=10.3,
    syn_planted_pre_t=-8.29,
    syn_null_true=0.0, syn_null_got=8.05, syn_null_maxerr=9.8, syn_null_pre_t=-1.46,
)


HEADER = f"""# Study 952 — After-Tax Equivalent 🏛️

**At what tax bracket do municipal bonds actually start beating taxable credit?**

Municipal bond coupons escape federal income tax. Every brochure turns that into the same
promise: above some marginal rate, munis are the better bond. This study measures the rate.

We pull each fund's daily closes **twice** — total return and price only — and recover the
monthly **income** leg as the difference, because income is the only leg the tax code
touches. Then we tax it: munis exempt from federal tax and the 3.8% NIIT surtax, taxable
credit paying `federal + NIIT + state`, T-bills state-exempt, the price leg left untaxed
(a buy-and-hold assumption we sweep). The after-tax difference turns out to be **exactly
linear in the bracket**, so the **break-even rate** solves in closed form.

Tape: **MUB, VTEB, SUB, HYD** (munis) against **AGG, LQD, VCIT** (taxable credit) and
**BIL** (cash), {R['start']} → {R['end']}, {R['n_months']} months.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the only
live cells run the offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Why a smaller coupon can still be the bigger cheque\n\n"
           f"Over the {R['n_mub_vcit']} months both funds existed, **MUB** (national "
           f"investment-grade munis) paid **{R['inc_mub_on_vcit']:.2f}%/yr** of income while "
           f"**VCIT** (intermediate investment-grade corporate bonds, a similar maturity) paid "
           f"**{R['inc_vcit']:.2f}%/yr**. The muni yields less — by design. The question is "
           f"whether the tax break more than closes that "
           f"{R['inc_vcit'] - R['inc_mub_on_vcit']:.2f} pp gap, and at what bracket."),
        code(
            ("R = %r\n" % ({"inc_mub": R["inc_mub_on_vcit"], "inc_vcit": R["inc_vcit"]},))
            + "gap = R['inc_vcit'] - R['inc_mub']\n"
              "print('MUB  (muni)  income %.2f%%/yr  - federally tax-free' % R['inc_mub'])\n"
              "print('VCIT (corp)  income %.2f%%/yr  - fully taxable' % R['inc_vcit'])\n"
              "print('pre-tax yield give-up: %.2f pp' % gap)\n"
              "print('so the muni needs a tax rate of about %.1f%% just to draw level on income'\n"
              "      % (100 * gap / R['inc_vcit']))"
        ),
        md("> 🔬 **For the quants.** The income leg is not quoted anywhere — we reconstruct it as "
           "`total return − price return` from two separate `yfinance` pulls (`auto_adjust` on "
           "and off). Ten of 199 months come out very slightly negative, an artefact of "
           "ex-dividend dates straddling a month end; flooring them at zero moves the mean by "
           "less than 0.001 bps/month and changes no conclusion."),
        md("## 2. The bracket where the two lines cross — and how blurry that line is\n\n"
           f"Because only the income leg is taxed, the after-tax difference is a straight line "
           f"in the tax rate. Solve for where it hits zero and you get the **break-even "
           f"bracket**. There are two honest versions of that number, and they are not the "
           f"same question.\n\n"
           f"**(a) Compare the coupons.** How high must your rate be before the muni's realised "
           f"*income stream* is worth more than the corporate one? Distribution streams are "
           f"smooth, so this is measured tightly:\n\n"
           f"| Muni | vs taxable | Income-leg break-even | 95% range |\n|---|---|--:|--:|\n"
           f"| MUB | VCIT | **{R['ibe_mub_vcit']:.1f}%** | "
           f"[{R['ibeci_mub_vcit'][0]:.1f}%, {R['ibeci_mub_vcit'][1]:.1f}%] |\n"
           f"| MUB | LQD | **{R['ibe_mub_lqd']:.1f}%** | "
           f"[{R['ibeci_mub_lqd'][0]:.1f}%, {R['ibeci_mub_lqd'][1]:.1f}%] |\n"
           f"| VTEB | VCIT | **{R['ibe_vteb_vcit']:.1f}%** | "
           f"[{R['ibeci_vteb_vcit'][0]:.1f}%, {R['ibeci_vteb_vcit'][1]:.1f}%] |\n\n"
           f"Call it **27–34%**, give or take three points. That is below the top two US "
           f"brackets (35.8% and 40.8% with the surtax), and it is this study's one solid "
           f"result.\n\n"
           f"**(b) Compare the whole return.** Do the same thing on *total* returns, so the "
           f"price legs count too, and the point estimate rises to **{R['be_mub_vcit']:.1f}%** "
           f"(MUB/VCIT). Tempting — but resample the tape and the honest range is "
           f"**[{R['beci_mub_vcit'][0]:.1f}%, {R['beci_mub_vcit'][1]:.1f}%]**. "
           f"{R['p_above_top_mub_vcit']*100:.0f}% of resamples say *no US bracket is high "
           f"enough*; {R['p_below_zero_mub_vcit']*100:.0f}% say the muni already wins with no "
           f"tax at all. The extra width is entirely the price legs, whose difference is "
           f"{R['price_leg_mub_vcit']:+.2f} bps/month at *t* = "
           f"{R['price_leg_mub_vcit_t']:+.2f} — i.e. nothing.\n\n"
           f"So the difference between (a) and (b) is not a finding. It is noise wearing a "
           f"decimal point."),
        md("## 3. The trap: why a *t*-stat here is not what it looks like\n\n"
           f"Race MUB against **AGG** (the US Aggregate index fund) and the muni wins by "
           f"**{R['d_mub_agg']:+.2f} bps/month** at the top bracket — *t* = "
           f"**{R['t_mub_agg']:+.2f}**, the one result on our tape that looks statistically "
           f"solid. It is not.\n\n"
           f"The after-tax gap is `pre-tax gap + tax break`. The **tax break is a coupon "
           f"stream**: big and almost the same every month. The **pre-tax gap** is the "
           f"difference of two bond-fund returns: small and wildly noisy. Add a near-constant "
           f"to a noisy series and the average jumps while the wobble does not — so the "
           f"*t*-stat rises **by arithmetic**, with no new information anywhere.\n\n"
           f"For MUB vs AGG the tax break supplies **{R['dec_mub_agg'][6]:.1f}%** of the average "
           f"and **{R['dec_mub_agg'][7]:.2f}%** of the wobble. Strip it out and the pre-tax gap "
           f"is {R['dec_mub_agg'][0]:+.2f} bps/month at *t* = {R['dec_mub_agg'][1]:+.2f}: "
           f"nothing. The same trick works on our synthetic *twin* world, where two identical "
           f"bonds cross *t* = 2 as the bracket rises with nothing planted at all.\n\n"
           f"**Every** row on this tape that clears the significance bar clears it this way. A "
           f"*t*-stat built from a tax constant tests whether the tax code is nonzero. It is."),
        md("## 4. So what happens above the break-even?\n\n"
           f"Almost nothing you could measure. At a 40.8% effective rate MUB beats VCIT by "
           f"**{R['ann_mub_vcit']:+.2f} pp/yr** — a *t* of **{R['t_mub_vcit']:+.2f}**, with a "
           f"bootstrap range of [{R['ci_mub_vcit_lo']:+.1f}, {R['ci_mub_vcit_hi']:+.1f}] bps/month "
           f"that comfortably contains zero. Split the sample at 2017 and the sign even flips "
           f"({R['era_vcit_e']:+.2f} then {R['era_vcit_l']:+.2f} bps/month). The honest reading is "
           f"that the muni market has **priced the tax break away** to about the level of the "
           f"top brackets, leaving a dead heat for the people it was supposedly free money for."),
        md("## 5. What the tax knobs are worth (all of them are assumptions)\n\n"
           f"None of the tax rates is data — they are imposed. So we swept them all. Adding a "
           f"California-scale state tax (13.3%) moves the MUB-vs-VCIT edge from "
           f"{R['state_none']:+.2f} to {R['state_ca_out']:+.2f} bps/month, because a *national* "
           f"muni fund's income is mostly out-of-state and gets taxed by your state too. Only a "
           f"**single-state** fund helps meaningfully ({R['state_ca_in']:+.2f} bps/month, still "
           f"*t* = {R['state_ca_in_t']:+.2f}). Taxing the price leg at 23.8% instead of leaving "
           f"it unrealised moves it to {R['cg_238']:+.2f}. No knob rescues the comparison."),
        md("## 6. Live check — the machinery is honest (offline synthetic)\n\n"
           "We build a fake bond world where we *know* the answer: the taxable leg is planted "
           "150 bp richer pre-tax, so the true break-even is exactly 33.3%. The solver has to "
           "find it from the tape alone. Then we plant twins — identical yields — and check the "
           "solver reports a break-even of zero and no pre-tax difference."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from after_tax import data, strategy as st\n"
            "for ss, label in [(1.0, 'planted 150 bp gap'), (0.0, 'twins (the null)')]:\n"
            "    panel, truth = data.synthetic_panel(signal_strength=ss, seed=952)\n"
            "    d = st.synthetic_detect(panel)\n"
            "    print('%-20s true break-even %6.2f%%  ->  recovered %6.2f%%   pre-tax diff t=%+.2f'\n"
            "          % (label, truth['planted_breakeven']*100, d['breakeven']*100, d['pretax_t']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** One number here is solid and it is not a win: on realised "
           f"coupons the muni needs about **{R['ibe_mub_vcit']:.1f}%** "
           f"([{R['ibeci_mub_vcit'][0]:.1f}%, {R['ibeci_mub_vcit'][1]:.1f}%]) against "
           f"like-for-like corporate credit — below the top brackets. Everything past that "
           f"dissolves: on *total* returns the crossover could be anywhere from "
           f"{R['beci_mub_vcit'][0]:.0f}% to {R['beci_mub_vcit'][1]:.0f}%; the top-bracket edge "
           f"is {R['ann_mub_vcit']:+.2f} pp/yr at *t* = {R['t_mub_vcit']:+.2f}; and every "
           f"significant-looking *t* on this page is the tax constant, not a market.\n"
           f"- **Tradability — Fragile.** The asset-location call is real and costs essentially "
           f"nothing (one round trip, 0.03 bps/month amortised): in a top bracket, hold the muni "
           f"fund in the taxable account. But the total-return margin is inside the noise, the "
           f"answer moves with a bracket you *assume*, and as a long-short trade a mere "
           f"25 bps/yr of borrow turns it negative.\n"
           f"- **The plain-words version.** The tax break on munis is real, and the market has "
           f"already priced it in to roughly the level of the top US brackets. If you are in one "
           f"of those brackets, hold munis in your taxable account — the coupon comparison says "
           f"so clearly. Just do not expect the tape to pay you for noticing: expect a tie."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 952 — After-Tax Equivalent — the teardown\n\n"
           "The linear-in-tau identity and the closed-form break-even, the five-bracket ladder, "
           "HAC *t* and block-bootstrap CIs, the era cut, the state / capital-gains / cost / "
           "borrow sweeps, the single-lag overlay, and the live synthetic control. Every "
           "real-tape number is frozen from `docs/results.md` (Fingerprint `%s`), monthly "
           "%s → %s. Every tax rate is a labelled PROXY."
           % (R["fp"], R["start"], R["end"])),
        code("R = %r" % (R,)),
        md("## The accounting, in one line\n\n"
           "`after_tax = price x (1 - capgain) + income x (1 - tau_income)`, with the income "
           "leg reconstructed as `total_return - price_return` per month. `tau_income` is 0 for "
           "munis federally (state only, and only on the out-of-state slice), `fed + NIIT + "
           "state` for taxable credit, `fed + NIIT` for T-bills. With the default "
           "`capgain = 0` this collapses to `total - tau x income`, so the after-tax difference "
           "between a muni and a taxable leg is **exactly affine in tau**:\n\n"
           "`d(tau) = d(0) + tau x mean(income_taxable)`\n\n"
           "which is why the break-even needs no search: "
           "`tau* = -mean(d(0)) / mean(income_taxable)`."),
        md("> 💡 **In plain words.** Raise the tax rate and the only thing that changes is how "
           "much the taxable bond's coupon is docked. That makes the muni-minus-taxable gap a "
           "straight line, and the break-even is just where the line crosses zero."),
        md("## Break-even by pairing — with the interval that decides whether to quote it\n\n"
           "`tau*` is a **ratio of two sample means**, and its numerator (the pre-tax "
           "total-return difference) never clears |*t*| ≥ 2 in any pairing on this tape. The "
           "point estimate is therefore far more precise-looking than the data warrants. The "
           "block bootstrap below is the correction; `p>40.8%` is the share of resamples in "
           "which **no US bracket is high enough**."),
        code(
            "rows = [('MUB','VCIT', R['pre_mub_vcit'], R['pre_mub_vcit_t'], R['be_mub_vcit'], R['beci_mub_vcit']),\n"
            "        ('MUB','LQD ', R['pre_mub_lqd'],  R['pre_mub_lqd_t'],  R['be_mub_lqd'],  R['beci_mub_lqd']),\n"
            "        ('VTEB','VCIT', None, None, R['be_vteb_vcit'], R['beci_vteb_vcit']),\n"
            "        ('MUB','AGG ', R['pre_mub_agg'],  R['pre_mub_agg_t'],  R['be_mub_agg'],  R['beci_mub_agg']),\n"
            "        ('VTEB','AGG ', None, None, R['be_vteb_agg'], R['beci_vteb_agg']),\n"
            "        ('SUB','BIL ', None, None, R['be_sub_bil'], R['beci_sub_bil']),\n"
            "        ('HYD','LQD ', None, None, R['be_hyd_lqd'], R['beci_hyd_lqd'])]\n"
            "for a, b, pre, t, be, ci in rows:\n"
            "    pre_s = '   n/a        ' if pre is None else 'pre-tax %+6.2f bps (t=%+5.2f)' % (pre, t)\n"
            "    print('%-5s vs %-5s  %s  tau* %+6.1f%%  95%% CI [%+7.1f%%, %+6.1f%%]'\n"
            "          % (a, b, pre_s, be, ci[0], ci[1]))\n"
            "print()\n"
            "print('MUB/VCIT: %.0f%% of resamples put tau* ABOVE the top US rate (40.8%%),'\n"
            "      % (100*R['p_above_top_mub_vcit']))\n"
            "print('          %.0f%% put it BELOW zero. The total-return break-even is not identified.'\n"
            "      % (100*R['p_below_zero_mub_vcit']))\n"
            "print('The width is the PRICE legs: MUB-VCIT price diff %+.2f bps/mo (t=%+.2f).'\n"
            "      % (R['price_leg_mub_vcit'], R['price_leg_mub_vcit_t']))"
        ),
        md("> ⚠️ **The sweep fallacy.** `tau*` is invariant to the bracket, the state rate, the "
           "in-state share and the capital-gains rate — because *no tax parameter enters its "
           "numerator*. \"Stable under every sweep\" is a statement about the arithmetic and "
           "carries **zero** information about precision. Only resampling the tape does "
           "(`strategy.breakeven_ci`, regression-tested both ways)."),
        md("## The half the tape does pin down: the income-leg break-even\n\n"
           "Drop the price legs and ask `tau* = 1 - y_muni / y_taxable` on realised monthly "
           "distributions net of fund fees. Income streams are smooth, so this is tight — and "
           "it is the study's one durable number."),
        code(
            "for a, b, ibe, ci in [('MUB','VCIT', R['ibe_mub_vcit'], R['ibeci_mub_vcit']),\n"
            "                      ('MUB','LQD ', R['ibe_mub_lqd'],  R['ibeci_mub_lqd']),\n"
            "                      ('VTEB','VCIT', R['ibe_vteb_vcit'], R['ibeci_vteb_vcit']),\n"
            "                      ('MUB','AGG ', R['ibe_mub_agg'],  R['ibeci_mub_agg'])]:\n"
            "    print('%-5s vs %-5s  income-leg tau* %+6.1f%%   95%% CI [%+5.1f%%, %+5.1f%%]'\n"
            "          % (a, b, ibe, ci[0], ci[1]))\n"
            "print()\n"
            "print('Note MUB/AGG: on income alone AGG OUT-yields MUB (tau* = %+.1f%%, CI strictly'\n"
            "      % R['ibe_mub_agg'])\n"
            "print('positive), so the %+.1f%% total-return break-even that made MUB look like a'\n"
            "      % R['be_mub_agg'])\n"
            "print('pre-tax winner is produced entirely by a price-leg diff of %+.2f bps/mo (t=%+.2f).'\n"
            "      % (R['price_leg_mub_agg'], R['price_leg_mub_agg_t']))"
        ),
        md("## Where the *t*-stat comes from — `d(tau) = d(0) + tau x i_taxable`\n\n"
           "The second term is a coupon stream: large in the mean, near-constant in time. "
           "Adding a near-deterministic constant to a noisy series lifts the mean without "
           "lifting the variance, so **HAC *t* climbs with the bracket by construction**."),
        code(
            "print('pair        pre-tax (t)      + tax term (sd)     = total (t)      mean%   var%')\n"
            "for name, d in [('MUB-AGG ', R['dec_mub_agg']), ('VTEB-AGG', R['dec_vteb_agg']),\n"
            "                ('SUB-BIL ', R['dec_sub_bil']), ('HYD-LQD ', R['dec_hyd_lqd']),\n"
            "                ('MUB-VCIT', R['dec_mub_vcit'])]:\n"
            "    print('%s  %+6.2f (%+5.2f)   %+6.2f (sd %4.2f)   %+6.2f (%+5.2f)  %6s  %5.2f'\n"
            "          % (name, d[0], d[1], d[2], d[3], d[4], d[5],\n"
            "             '--' if d[6] is None else '%.1f' % d[6], d[7]))\n"
            "print()\n"
            "print('Every row crossing |t| = 2 does so on a term carrying < 2% of the variance.')\n"
            "print('That is a test of whether the tax code is nonzero, not of a market.')"
        ),
        md("## The bracket ladder — MUB vs VCIT (like-for-like) and MUB vs AGG (not)"),
        code(
            "print('rate   MUB-VCIT bps (t)      MUB-AGG bps (t)')\n"
            "for r, v, vt, g, gt in zip(R['ladder_rates'], R['ladder_vcit'], R['ladder_vcit_t'],\n"
            "                           R['ladder_agg'], R['ladder_agg_t']):\n"
            "    print('%5.1f%%  %+7.2f (%+5.2f)      %+7.2f (%+5.2f)' % (r, v, vt, g, gt))\n"
            "print()\n"
            "print('MUB-VCIT at 40.8%%: %+.2f bps/mo (%+.2f pp/yr)  HAC t=%+.2f  boot CI [%+.1f, %+.1f]'\n"
            "      % (R['d_mub_vcit'], R['ann_mub_vcit'], R['t_mub_vcit'],\n"
            "         R['ci_mub_vcit_lo'], R['ci_mub_vcit_hi']))\n"
            "print('MUB-AGG  at 40.8%%: %+.2f bps/mo (%+.2f pp/yr)  HAC t=%+.2f  boot CI [%+.1f, %+.1f]'\n"
            "      % (R['d_mub_agg'], R['ann_mub_agg'], R['t_mub_agg'],\n"
            "         R['ci_mub_agg_lo'], R['ci_mub_agg_hi']))\n"
            "print('excess-of-after-tax-cash Sharpe, MUB vs VCIT: %+.2f vs %+.2f'\n"
            "      % (R['sh_mub_vcit_m'], R['sh_mub_vcit_t']))"
        ),
        md("The MUB-AGG ladder is the mechanism above in motion: *t* climbs monotonically from "
           f"{R['ladder_agg_t'][0]:+.2f} to {R['ladder_agg_t'][-1]:+.2f} purely because the "
           f"bracket multiplies a near-constant coupon stream. The pairings that clear "
           f"|*t*| ≥ 2 on the pooled sample — MUB/AGG ({R['t_mub_agg']:+.2f}), VTEB/AGG "
           f"({R['t_vteb_agg']:+.2f}), SUB/BIL ({R['t_sub_bil']:+.2f}) — all do so on that "
           "constant. Reading the bottom row of a bracket ladder as a significance test is the "
           "specific mistake this study makes easy."),
        md("## Era cut (split 2017-01) — nothing survives in halves"),
        code(
            "print('MUB-VCIT  pre-2017 (n=%3d): %+6.2f bps  t=%+5.2f' % (R['era_vcit_e_n'], R['era_vcit_e'], R['era_vcit_e_t']))\n"
            "print('MUB-VCIT  2017-on  (n=%3d): %+6.2f bps  t=%+5.2f   <- sign flips' % (R['era_vcit_l_n'], R['era_vcit_l'], R['era_vcit_l_t']))\n"
            "print('MUB-AGG   pre-2017 (n=%3d): %+6.2f bps  t=%+5.2f' % (R['era_agg_e_n'], R['era_agg_e'], R['era_agg_e_t']))\n"
            "print('MUB-AGG   2017-on  (n=%3d): %+6.2f bps  t=%+5.2f   <- neither half clears 2' % (R['era_agg_l_n'], R['era_agg_l'], R['era_agg_l_t']))"
        ),
        md("## The PROXY sweeps — state, capital gains, income floor"),
        code(
            "print('state 0%%             : %+5.2f bps/mo' % R['state_none'])\n"
            "print('state 13.3%%, national: %+5.2f bps/mo   (income is mostly out-of-state -> taxed anyway)' % R['state_ca_out'])\n"
            "print('state 13.3%%, in-state: %+5.2f bps/mo (t=%+.2f)  <- best case, still nowhere near 2' % (R['state_ca_in'], R['state_ca_in_t']))\n"
            "print()\n"
            "print('cap-gains  0%% (unrealised, default): %+5.2f bps/mo' % R['cg_0'])\n"
            "print('cap-gains 23.8%% (20%% + NIIT)       : %+5.2f bps/mo' % R['cg_238'])\n"
            "print()\n"
            "print('income floor on : %+6.3f bps/mo' % R['floor_on'])\n"
            "print('income floor off: %+6.3f bps/mo   <- the reconstruction choice is immaterial' % R['floor_off'])"
        ),
        md("## Costs (irrelevant) and borrow (fatal)\n\n"
           "As an asset-location choice the friction is a single round trip out of the "
           "incumbent, amortised. As a long-short spread the short taxable leg pays borrow — "
           "and that is what kills it."),
        code(
            "print('cost  0 bps one-way: %+5.2f bps/mo' % R['cost0'])\n"
            "print('cost 25 bps one-way: %+5.2f bps/mo   <- one trade in 17 years' % R['cost25'])\n"
            "print()\n"
            "print('borrow   0 bps/yr: MUB-VCIT %+6.2f   MUB-AGG %+6.2f' % (R['borrow0_vcit'], R['borrow0_agg']))\n"
            "print('borrow  25 bps/yr: MUB-VCIT %+6.2f   MUB-AGG %+6.2f (t=%+.2f)  <- both broken'\n"
            "      % (R['borrow25_vcit'], R['borrow25_agg'], R['borrow25_agg_t']))\n"
            "print('borrow 100 bps/yr: MUB-VCIT %+6.2f   MUB-AGG %+6.2f' % (R['borrow100_vcit'], R['borrow100_agg']))"
        ),
        md("> 💡 **In plain words.** Choosing which bond fund to own costs one trade, so friction "
           "is a rounding error. Trying to *arbitrage* the pair means shorting the taxable "
           "fund, and the borrow fee on a corporate-bond ETF is larger than the entire edge."),
        md("## The one arm with a signal — and therefore exactly one execution lag\n\n"
           "Trailing 12-month after-tax income yields known at the close of month *t* decide "
           "the leg held in month *t+1*. 3 bps one-way × NAV per switch."),
        code(
            "print('MUB/VCIT overlay: n=%d  in-muni %.1f%%  switches %d over ~17 years'\n"
            "      % (R['ov_n'], R['ov_in'], R['ov_switches']))\n"
            "print('  overlay %+.2f%%/yr vs simply holding the muni %+.2f%%/yr'\n"
            "      % (R['ov_ann'], R['ov_hold']))\n"
            "print('  edge %+.2f bps/mo (HAC t=%+.2f) -> the bracket question has no timing dimension'\n"
            "      % (R['ov_edge'], R['ov_t']))"
        ),
        md("## Live synthetic control — machinery proof, never a stamp\n\n"
           "Forty years of monthly data, eight seeds. **Planted world:** the taxable leg yields "
           "150 bp more pre-tax, so the true break-even is exactly 33.33% and the after-tax "
           "difference must flip sign across it. **Twin null:** identical yields and duration, "
           "so the break-even must collapse to ~0 and no *pre-tax* difference may be "
           "manufactured.\n\n"
           "Note what the null deliberately does *not* claim: on twins the *after-tax* "
           "difference at a positive bracket is large and significant, because it is pure "
           "arithmetic (`tau × yield`). That is the effect being measured; the control's job is "
           "to show the machinery attributes none of it to a market edge."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from after_tax import data, strategy as st\n"
            "for ss, label in [(1.0, 'planted 150bp gap'), (0.0, 'twin null      ')]:\n"
            "    errs = []\n"
            "    for s in range(8):\n"
            "        panel, truth = data.synthetic_panel(signal_strength=ss, seed=952 + s)\n"
            "        errs.append(st.breakeven_rate(panel, 'muni', 'taxable')['breakeven']\n"
            "                    - truth['planted_breakeven'])\n"
            "    panel, truth = data.synthetic_panel(signal_strength=ss, seed=952)\n"
            "    d = st.synthetic_detect(panel)\n"
            "    errs = np.array(errs)\n"
            "    print('%s true %6.2f%%  recovered %6.2f%%  (8 seeds: mean err %+.2f pp, max |err| %.2f pp)'\n"
            "          % (label, truth['planted_breakeven']*100, d['breakeven']*100,\n"
            "             errs.mean()*100, np.abs(errs).max()*100))\n"
            "    print('    pre-tax diff %+6.2f bps (t=%+5.2f)   below break-even %+6.2f bps -> above %+6.2f bps'\n"
            "          % (d['pretax_diff_bps'], d['pretax_t'],\n"
            "             d['diff_below_breakeven'], d['diff_above_breakeven']))"
        ),
        md("### The t-stat demonstration, live on the twin null\n\n"
           "Two **statistically identical** bonds, nothing planted. Watch the after-tax HAC *t* "
           "walk past 2 as the assumed bracket rises. Nothing arrives but arithmetic — which is "
           "why no |*t*| ≥ 2 in the real-tape tables above can be read as a market effect."),
        code(
            "panel, _ = data.synthetic_panel(signal_strength=0.0, seed=952)\n"
            "print('twin null: identical yields, identical duration, zero planted edge')\n"
            "for f in (0.0, 0.10, 0.20, 0.30, 0.40):\n"
            "    r = st.race(panel, 'muni', 'taxable', st.tax_profile(f, 0.0, 0.0),\n"
            "                cash='cash', cost_bps=0.0)\n"
            "    print('  bracket %4.0f%%   after-tax diff %+7.2f bps/mo   HAC t = %+5.2f'\n"
            "          % (100*f, r['diff_bps'], r['t_diff']))\n"
            "dec = st.tax_constant_decomposition(panel, 'muni', 'taxable', st.tax_profile(0.37))\n"
            "print('\\nat 40.8%%: tax term is %.1f%% of the mean and %.2f%% of the variance'\n"
            "      % (100*dec['mean_share_tax'], 100*dec['var_share_tax']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** Exactly one estimand here is identified, and it is not an "
           f"outperformance. The **income-leg break-even** is tight — "
           f"**{R['ibe_mub_vcit']:.1f}%** "
           f"[{R['ibeci_mub_vcit'][0]:.1f}, {R['ibeci_mub_vcit'][1]:.1f}] vs VCIT, "
           f"**{R['ibe_mub_lqd']:.1f}%** "
           f"[{R['ibeci_mub_lqd'][0]:.1f}, {R['ibeci_mub_lqd'][1]:.1f}] vs LQD — measured on "
           f"realised distributions net of fees, and it sits below the top two brackets. "
           f"Everything built on top of it fails. The **total-return break-even is not "
           f"identified**: {R['be_mub_vcit']:.1f}% point, 95% CI "
           f"[{R['beci_mub_vcit'][0]:.1f}%, {R['beci_mub_vcit'][1]:.1f}%], "
           f"{R['p_above_top_mub_vcit']*100:.0f}% of draws above the top US rate — its numerator "
           f"never clears |*t*| ≥ 2 in any pairing (largest {R['pre_mub_vcit_t']:+.2f}). The "
           f"**after-tax race is a dead heat**: {R['d_mub_vcit']:+.2f} bps/mo "
           f"({R['ann_mub_vcit']:+.2f} pp/yr), HAC *t* = {R['t_mub_vcit']:+.2f}, CI "
           f"[{R['ci_mub_vcit_lo']:+.1f}, {R['ci_mub_vcit_hi']:+.1f}]. And the three pairings "
           f"clearing |*t*| ≥ 2 clear it on a **tax constant** worth "
           f"{R['dec_mub_agg'][6]:.0f}% of the mean and {R['dec_mub_agg'][7]:.2f}% of the "
           f"variance — the same lift appears on the synthetic twin null with nothing planted. "
           f"**No |*t*| ≥ 2 in this study is evidence of a market effect**, and none survives "
           f"either era half. The control recovers a planted 33.33% break-even to within "
           f"{R['syn_planted_maxerr']:.1f} pp across 8 seeds and stays quiet on twins "
           f"(pre-tax *t* = {R['syn_null_pre_t']:+.2f}), so the dead heat is the muni market, "
           f"not the harness.\n"
           f"- **Tradability — Fragile.** Real and free as an *allocation* (one round trip, "
           f"0.03 bps/mo amortised; in a top bracket hold munis in the taxable account, which "
           f"the tight income-leg break-even does support), but the total-return margin sits "
           f"inside the noise, the answer rides on an assumed bracket, the crossover cannot be "
           f"pinned better than [{R['beci_mub_vcit'][0]:.0f}%, {R['beci_mub_vcit'][1]:.0f}%], "
           f"and as a *trade* {R['borrow25_vcit']:+.2f} bps/mo at 25 bps/yr borrow is the whole "
           f"story."),
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
