"""Generate the two narrative notebooks for Study 961 (Which Gold).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
synthetic control, which is clearly labelled as synthetic and never sits under a real-tape
banner.
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
# Frozen real-tape headline — mirror of docs/results.md.
# GLD / IAU / GLDM / SGOL / BAR, daily adjusted closes, 2018-06-26 -> 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2018-06-26", end="2026-06-30", n_days=2013, n_months=96, fp="9af52c692ea9",
    # fee sheet (ASSUMPTION) and realised cohort-relative tracking difference
    fees={"GLD": 40.00, "IAU": 25.00, "GLDM": 10.00, "SGOL": 17.00, "BAR": 17.49},
    td={"GLD": -16.75, "IAU": -2.24, "GLDM": 9.43, "SGOL": 5.41, "BAR": 4.15},
    # t_naive is the CONSERVATIVE one on this mean-reverting tape; HAC is quoted beside it
    td_t={"GLD": -3.02, "IAU": -0.31, "GLDM": 1.80, "SGOL": 0.56, "BAR": 0.54},
    td_t_hac={"GLD": -6.95, "IAU": -0.62, "GLDM": 3.45, "SGOL": 1.04, "BAR": 1.02},
    td_up={"GLD": 32, "IAU": 47, "GLDM": 59, "SGOL": 49, "BAR": 51},
    # the rank test
    spearman=-1.0000, p_perm=0.0167, p_floor=0.0167, n_perm=120,
    pt_slope=-0.894, pt_r2=0.992, pt_slope_blend=-0.943, pt_r2_blend=0.999,
    # the headline pair — pure-tape measurement, pair selected off the fee sheet
    pair="GLDM - GLD", spread=26.17, spread_t=3.18, spread_t_hac=6.32, spread_up=65,
    spread_sd_m=6.7, spread_acf1=-0.424, pos_years=9, n_years=9,
    ci_lo=9.54, ci_hi=41.88, ci_neg=0.06,               # iid bootstrap — the conservative CI
    ci_blk_lo=17.09, ci_blk_hi=35.28, ci_blk_neg=0.0,   # moving-block — inherits mean reversion
    hac_sweep=[("naive", 3.18), ("lag 1", 4.22), ("lag 3", 6.32), ("lag 12", 9.43)],
    daily=26.36, daily_t=2.19, daily_t_naive=0.78, daily_sd=5.98, daily_acf1=-0.462,
    endpoint=26.28, fee_gap=30.00, fee_gap_blend=27.74,
    # measurement floor
    floor_bpy=22.5, floor_sd_d=7.43, floor_sd_m=9.2,
    # eras and the fee-stable sub-window (naive t, HAC beside)
    era_e_n=42, era_e=20.98, era_e_t=1.43, era_e_t_hac=2.87, era_e_up=23,
    era_l_n=54, era_l=30.21, era_l_t=3.28, era_l_t_hac=6.62, era_l_up=42,
    stable_n=65, stable=26.66, stable_t=2.69, stable_t_hac=5.24, stable_up=47,
    stable_rho=-1.0000,
    # what it is worth
    cost1=0.262, cost5=1.302, cost10=2.587, cost20=5.107, per100k10=2587,
    # ownership race
    sh_cheap=0.7799, sh_dear=0.7613, sh_chase=0.7762,
    cagr_cheap=15.476, cagr_dear=15.172, cagr_chase=15.427,
    sh_gap=0.0187, cd_t=0.74, cd_t_hac=2.06, chase_t=-0.14, chase_t_hac=-0.41, n_switches=4,
    # counterweights
    adv={"GLD": 4085, "IAU": 654, "GLDM": 433, "SGOL": 175, "BAR": 21},
    adv_days={"GLD": 0.02, "IAU": 0.15, "GLDM": 0.23, "SGOL": 0.57, "BAR": 4.80},
    be5=69.7, be10=139.5, be30=418.4,
    tax_20_50=26.4, tax_238_300=75.1,
    ls_gross=18.17, ls_dead_above=18.0,
    # synthetic control
    syn_gap=30.0, syn_rec=30.98, syn_t=2.55, syn_rho=-1.000, syn_slope=-1.041,
    syn_null_mean=-0.47, syn_null_sd=1.07, syn_null_fire=0, syn_null_p=0,
)

ORDER = ["GLDM", "SGOL", "BAR", "IAU", "GLD"]


HEADER = f"""# Study 961 — Which Gold 🪙

**Five wrappers, one metal. Does the cheap one actually keep more of it?**

**GLD, IAU, GLDM, SGOL, BAR** all hold allocated London bullion in a vault, publish a bar
list, and strike at the same 16:00 New York close. They differ in one contractual respect:
the sponsor fee, from **10 bp** (GLDM) to **40 bp** (GLD). If the fee sheet is a real number
rather than a marketing one, the ranking of what these five actually delivered should be the
exact reverse of the ranking of what they charge.

Common window **{R['start']} → {R['end']}** ({R['n_days']:,} sessions, {R['n_months']}
complete months). None of the five pays a distribution, so their adjusted close *is* their
price close.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the live
cells run the offline synthetic control and are labelled as such. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The setup — the only thing that separates them is the price tag\n\n"
           "There is no skill question here, and no market view. Nobody is forecasting gold. "
           "The five trusts own the same metal; the sponsor takes its cut by quietly selling "
           "a sliver of bullion. So the question is arithmetic: **does the sliver show up?**"),
        code(
            "R = %r\nORDER = %r\n"
            "print('wrapper   fee (bp/yr)')\n"
            "for t in ORDER:\n"
            "    print('  %%-6s %%6.2f' %% (t, R['fees'][t]))\n"
            "print('\\nfee spread across the cohort: %%.0f bp/yr' %% (max(R['fees'].values()) - min(R['fees'].values())))"
            % ({"fees": R["fees"]}, ORDER)
        ),
        md("## 2. The answer — the ranking is perfect\n\n"
           "Below: each wrapper's realised return minus the average of all five, in basis "
           "points a year, over eight years. Read the two columns side by side."),
        code(
            "R = %r\nORDER = %r\n"
            "print('wrapper   fee    what it actually delivered vs the cohort average')\n"
            "for t in ORDER:\n"
            "    print('  %%-6s %%5.2f   %%+7.2f bp/yr   (t = %%+5.2f, up in %%d of 96 months)'\n"
            "          %% (t, R['fees'][t], R['td'][t], R['td_t'][t], R['td_up'][t]))\n"
            "print('\\nrank correlation between fee and outcome: %%.4f  (a perfect inversion)' %% R['spearman'])"
            % ({"fees": R["fees"], "td": R["td"], "td_t": R["td_t"], "td_up": R["td_up"],
                "spearman": R["spearman"]}, ORDER)
        ),
        md(f"Cheapest first, priciest last, in exactly the order the fee sheet predicts, with "
           f"**no inversions**. The gap between the ends — GLDM's 10 bp against GLD's 40 bp — "
           f"is worth **{R['spread']:+.2f} basis points a year** on the tape, against the "
           f"**{R['fee_gap']:.0f} bp** the fee sheets promise: *t* = **{R['spread_t']:+.2f}**, "
           f"positive in **{R['spread_up']}** of {R['n_months']} months and in "
           f"**{R['pos_years']} of {R['n_years']}** calendar years, with a 95% interval "
           f"[{R['ci_lo']:+.1f}, {R['ci_hi']:+.1f}] that never touches zero.\n\n"
           f"Note how wide that interval is. The tape says *somewhere between 10 and 42 basis "
           f"points* — not 26 on the nose. What it does not say is zero.\n\n"
           "> 🔬 **For the quants:** the estimator is the mean of non-overlapping "
           "complete-calendar-month log differences, annualised. The *t* above is the plain "
           "iid one **on purpose**: this residual mean-reverts "
           f"(acf1 = {R['spread_acf1']:+.2f} monthly, {R['daily_acf1']:+.2f} daily), so a "
           f"Newey-West *t* comes out *larger* ({R['spread_t_hac']:+.2f}) rather than smaller, "
           "and larger still as you widen the bandwidth. We quote the small one. The daily "
           f"view of the same effect is powerless either way: naive *t* = "
           f"{R['daily_t_naive']:+.2f}."),
        md("## 3. What it costs to own the wrong one\n\n"
           "No turnover, no timing, no decision after the first one. The gap simply "
           "compounds."),
        code(
            "R = %r\n"
            "for y, pct in [(1, R['cost1']), (5, R['cost5']), (10, R['cost10']), (20, R['cost20'])]:\n"
            "    print('%%2d years of the dear wrapper: %%.3f%%%% of your gold sleeve, or $%%6s per $100,000'\n"
            "          %% (y, pct, format(int(round(pct/100*100000)), ',')))"
            % ({"cost1": R["cost1"], "cost5": R["cost5"], "cost10": R["cost10"], "cost20": R["cost20"]},)
        ),
        md(f"Ten years of holding GLD instead of GLDM costs about **{R['cost10']:.1f}% of the "
           f"sleeve** — roughly ${R['per100k10']:,} on $100,000. That is the whole prize. It is "
           "small, it is certain, and it requires nothing of you but reading a fee sheet once."),
        md("## 4. The catch — three of them\n\n"
           f"**(a) You cannot fix this by switching.** Selling an appreciated GLD position "
           f"triggers tax (US bullion trusts are taxed as collectibles). At a 20% rate on a "
           f"position that has doubled, the 26 bp gap needs **{R['tax_20_50']:.0f} years** to "
           f"repay the bill; at 23.8% on a position up 4×, **{R['tax_238_300']:.0f} years**. "
           "This is a rule for money you are *about to* invest.\n\n"
           f"**(b) Cheapest is not always buyable.** The fee-cheapest tier is also the "
           f"thinnest. A $100m ticket is {R['adv_days']['GLD']:.2f} days of GLD's volume and "
           f"**{R['adv_days']['BAR']:.1f} days** of BAR's. GLDM, at "
           f"${R['adv']['GLDM']:,}m a day, is the one place where cheap and liquid coincide.\n\n"
           f"**(c) The fine print of the ladder is unreadable.** The tape can separate a 30 bp "
           f"gap. It cannot separate a 7 bp one: GLDM against SGOL, SGOL against BAR — all "
           f"statistically nothing. The ranking is right at the top and noise at the rungs."),
        md("## 5. Chasing last year's best tracker does not work\n\n"
           "The obvious next thought — buy whichever wrapper tracked best last year — is worse "
           "than doing nothing. Ranked at each year-end, filled at the next session's close, "
           f"2 bp of cost per switch: **{R['sh_chase']:.4f}** excess Sharpe against "
           f"**{R['sh_cheap']:.4f}** for simply owning the cheapest ({R['n_switches']} switches "
           f"over eight years, *t* on the difference = {R['chase_t']:+.2f}). The fee is the "
           "signal; last year's tracking is not."),
        md("## 6. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "**This cell is synthetic, not the real tape.** On a *planted* world where five "
           "wrappers really are shaved by their published fees, the stack must recover the "
           "ladder. On a null where every wrapper charges the same fee — while the published "
           "sheet still shows dispersion — it must find nothing."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from which_gold import data, strategy as st\n"
            "pl_p, pl_t = data.synthetic_panel(signal_strength=1.0, seed=961)\n"
            "pl = st.synthetic_detect(pl_p, pl_t)\n"
            "print('SYNTHETIC planted (gap %.1f bp/yr): recovered %+.2f bp/yr, t=%+.2f, rank corr %+.3f'\n"
            "      % (pl['planted_gap_bpy'], pl['pair_spread_bpy'], pl['pair_t'], pl['spearman']))\n"
            "nl = [st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=961+s)) for s in range(8)]\n"
            "sp = np.array([n['pair_spread_bpy'] for n in nl]); tt = np.array([n['pair_t'] for n in nl])\n"
            "print('SYNTHETIC null x8: spread mean %+.2f bp/yr (sd %.2f), |t|>=2 in %d/8 (should be 0)'\n"
            "      % (sp.mean(), sp.std(ddof=1), int((abs(tt) >= 2).sum())))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The cheapest wrapper beat the priciest by "
           f"**{R['spread']:+.2f} bp/yr** with *t* = **{R['spread_t']:+.2f}**, an interval "
           f"[{R['ci_lo']:+.1f}, {R['ci_hi']:+.1f}] clear of zero, the same sign in "
           f"**{R['pos_years']} of {R['n_years']}** calendar years and in **both** halves of "
           f"the sample, and the five-wrapper ranking is a **perfect** inversion of the fee "
           f"ranking. Where the fee sheet is coarse the tape confirms it; where it is fine "
           f"(7 bp apart) the tape cannot see it — and one wrapper at a time, only the "
           f"priciest separates from the pack on its own.\n"
           f"- **Tradability — Investable.** It is a purchase decision, not a strategy: no "
           f"forecast, no turnover, no capacity limit on the wrapper that wins, and "
           f"**{R['cost10']:.1f}% of a ten-year gold sleeve** at stake. Buy the cheap one with "
           f"new money and never trade it — do not sell an appreciated holding to get there, "
           f"do not try to short the dear one, and do not chase last year's tracker."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 961 — Which Gold — the teardown\n\n"
           "Realised tracking differences on a five-wrapper, single-asset cross-section: the "
           "three estimators and why they disagree, the measurement floor, the fee-rank test "
           "against an exactly enumerated permutation null, the pass-through regression, the "
           "cheapest-minus-priciest spread with **naive and HAC *t*s and two bootstraps**, an "
           "era cut, a fee-stable sub-window, the excess-of-cash ownership race with one "
           "execution lag, four swept counterweights, and the live synthetic control.\n\n"
           "**Read the naive *t*, not the HAC one.** The wrapper-versus-wrapper residual "
           "mean-reverts at both frequencies, so Newey-West *shrinks* the long-run variance "
           "here and the HAC *t* runs about twice the iid one — and keeps growing with the "
           "bandwidth. Same logic on the interval: the block bootstrap inherits the mean "
           "reversion, so the **iid** bootstrap CI is the one quoted as the honest one.\n\n"
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`); the "
           "synthetic cells are labelled and never sit under a real-tape banner."
           % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The panel and its non-tape inputs\n\n"
           "> 💡 **In plain words:** the fee sheet is the only thing in this study that is not "
           "measured. Everything with a *t* next to it is pure tape — but the fee sheet still "
           "chose which pair to point the tape at.\n\n"
           "Daily adjusted closes, `auto_adjust=True`. None of the five grantor trusts "
           "distributes — they sell bullion to pay the sponsor — so **adjusted close = price "
           "close** and the total-return / price-only labels coincide. BIL is genuine total "
           "return and appears only as the cash leg. **Survivorship:** these are five wrappers "
           "that still exist; closed bullion vehicles are absent by construction, which "
           "flatters the cohort *level* and cannot manufacture the cross-sectional *ordering*.\n\n"
           "ASSUMPTIONS, all labelled and swept where they matter: *today's* published fee sheet "
           "(hindsight-carrying — on day one of the window GLDM charged 18 bp and was not the "
           "cheapest of the five), GLDM's 18 → 10 bp cut on 2020-10-01 (an announcement date, "
           "invisible on a price tape, and the only change this study records), the round-trip "
           "execution differential, the borrow rate, and the capital-gains rate. One "
           "**selection** on top of those: the headline pair is chosen off that sheet — the "
           "number is pure tape, the choice of legs is not."),
        md("## 2. The measurement floor — what this tape can and cannot resolve"),
        code(
            "print(f\"pairwise noise: {R['floor_sd_d']:.2f} bp/day, {R['floor_sd_m']:.1f} bp/month \"\n"
            "      f\"over {R['n_months']} months\")\n"
            "print(f\"smallest gap separable from zero at |t|=2: {R['floor_bpy']:.1f} bp/yr\")\n"
            "print(f\"published fee spread across the cohort:    {R['fee_gap']:.1f} bp/yr\")\n"
            "print('-> the coarse ranking is resolvable; the single-digit rungs are not')"
        ),
        md("## 3. Cohort-relative tracking difference, three estimators\n\n"
           "Each wrapper minus the equal-weight cohort mean: the bullion price and any common "
           "strike effect cancel exactly, leaving the fee plus each wrapper's own "
           "premium/discount."),
        code(
            "print(f\"{'fund':6s} {'fee':>6s} {'monthly':>9s} {'t':>7s} {'[HAC]':>8s} {'up':>8s}\")\n"
            "for t in %r:\n"
            "    print(f\"{t:6s} {R['fees'][t]:6.2f} {R['td'][t]:+9.2f} {R['td_t'][t]:+7.2f} \"\n"
            "          f\"{R['td_t_hac'][t]:+8.2f} {R['td_up'][t]:5d}/{R['n_months']}\")\n"
            "print(f\"\\ncross-section sums to ~0 by construction: {sum(R['td'].values()):+.2f} bp/yr\")\n"
            "print('on the honest t only GLD, the priciest, separates from the cohort mean alone;')\n"
            "print('GLDM is the right sign and size at t=+1.80 -- a 9 bp effect against a 22.5 bp')\n"
            "print('floor. The evidence is the widest PAIR and the JOINT ranking, not one wrapper.')"
            % (ORDER,)
        ),
        md("## 4. The fee-rank test and the pass-through regression\n\n"
           "> 💡 **In plain words:** the first asks whether the fee *order* predicts the outcome "
           "*order*; the second asks whether a basis point of fee costs a basis point of return."),
        code(
            "print(f\"Spearman(fee, tracking difference) = {R['spearman']:+.4f}\")\n"
            "print(f\"exact permutation p = {R['p_perm']:.4f} over all {R['n_perm']} permutations\")\n"
            "print(f\"  ...which is the HARD FLOOR: with 5 funds no match, however perfect,\")\n"
            "print(f\"     can print below p = {R['p_floor']:.4f}. The rank test is at its ceiling.\")\n"
            "print(f\"\\npass-through slope (today's sheet): {R['pt_slope']:+.3f}  R2 {R['pt_r2']:.3f}\")\n"
            "print(f\"pass-through slope (time-blended) : {R['pt_slope_blend']:+.3f}  R2 {R['pt_r2_blend']:.3f}\")\n"
            "print('-> a basis point of fee costs ~0.9 of a basis point of return, and the fee')\n"
            "print('   sheet explains 99% of a five-point cross-section')"
        ),
        md("## 5. The headline pair — GLDM (10 bp) minus GLD (40 bp)\n\n"
           "No fee number enters this measurement. **Which** two wrappers get differenced is "
           "read off the fee sheet, though — the pair is a selection, and it is the widest gap "
           "on the sheet, i.e. the pair with the most to find."),
        code(
            "print(f\"monthly (non-overlapping): {R['spread']:+.2f} bp/yr  naive t {R['spread_t']:+.2f}  \"\n"
            "      f\"[HAC {R['spread_t_hac']:+.2f}]  {R['spread_up']}/{R['n_months']} months up  \"\n"
            "      f\"sd {R['spread_sd_m']:.1f} bp/mo  acf1 {R['spread_acf1']:+.3f}\")\n"
            "print(f\"positive in {R['pos_years']}/{R['n_years']} calendar years\")\n"
            "print(f\"iid bootstrap   95% CI   : [{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}] bp/yr  \"\n"
            "      f\"share<0 {R['ci_neg']:.2f}%   <- the conservative interval\")\n"
            "print(f\"block bootstrap 95% CI   : [{R['ci_blk_lo']:+.2f}, {R['ci_blk_hi']:+.2f}] bp/yr  \"\n"
            "      f\"share<0 {R['ci_blk_neg']:.2f}%   (inherits the mean reversion)\")\n"
            "print(f\"daily                    : {R['daily']:+.2f} bp/yr  naive t {R['daily_t_naive']:+.2f}  \"\n"
            "      f\"[HAC {R['daily_t']:+.2f}]\")\n"
            "print(f\"endpoint                 : {R['endpoint']:+.2f} bp/yr\")\n"
            "print(f\"\\ndaily residual: sd {R['daily_sd']:.2f} bp, acf1 {R['daily_acf1']:+.3f}\")\n"
            "print('HAC bandwidth sweep: ' + '  '.join('%s=%+.2f' % (k, v) for k, v in R['hac_sweep']))\n"
            "print('-> the residual MEAN-REVERTS at BOTH frequencies, so the HAC long-run variance')\n"
            "print('   sits BELOW the iid one: the HAC t is the ANTI-conservative choice and it')\n"
            "print('   keeps climbing with the bandwidth. Monthly aggregation is the right')\n"
            "print('   construction (non-overlapping observations) but it is NOT a conservative')\n"
            "print('   one, and this study does not pretend otherwise: it quotes the naive t.')\n"
            "print(f\"\\nrealised {R['spread']:.2f} vs sheet gap {R['fee_gap']:.2f}; time-blending GLDM's\")\n"
            "print(f\"18 bp launch fee expects {R['fee_gap_blend']:.2f} -> about half the shortfall is fee\")\n"
            "print('history and the rest sits well inside the iid bootstrap half-width (+/-16 bp).')"
        ),
        md("## 6. Era cut and the fee-stable sub-window\n\n"
           "Two ways of asking whether this is one lucky stretch."),
        code(
            "print(f\"2018-07 -> 2021-12 (n={R['era_e_n']}): {R['era_e']:+.2f} bp/yr  \"\n"
            "      f\"t={R['era_e_t']:+.2f} [HAC {R['era_e_t_hac']:+.2f}]  {R['era_e_up']}/{R['era_e_n']} up\")\n"
            "print(f\"2022-01 -> 2026-06 (n={R['era_l_n']}): {R['era_l']:+.2f} bp/yr  \"\n"
            "      f\"t={R['era_l_t']:+.2f} [HAC {R['era_l_t_hac']:+.2f}]  {R['era_l_up']}/{R['era_l_n']} up\")\n"
            "print('-> positive in BOTH halves, significant on the honest t in the LATE one only.')\n"
            "print('   42 months of a ~21 bp effect against a 22.5 bp floor CANNOT be significant,')\n"
            "print('   so the era cut buys sign-stability, not two independent confirmations.')\n"
            "print('   The early era is smaller by about what GLDM 18 bp (until 2020-10-01)')\n"
            "print(f\"   implies, and the late era prints {R['era_l']:.2f} against a {R['fee_gap']:.2f} bp fee gap.\")\n"
            "print(f\"\\nfee-stable sub-window (2021-01 ->, n={R['stable_n']} months): \"\n"
            "      f\"{R['stable']:+.2f} bp/yr  t={R['stable_t']:+.2f} [HAC {R['stable_t_hac']:+.2f}]  \"\n"
            "      f\"{R['stable_up']}/{R['stable_n']} up  Spearman {R['stable_rho']:+.4f}\")\n"
            "print('-> the cut that matters most: no fee assumption does any work in it, and it')\n"
            "print('   clears |t|=2 on its own.')"
        ),
        md("## 7. The ownership race, excess-of-cash — one execution lag, costs × NAV\n\n"
           "> 💡 **In plain words:** 26 bp a year on a 17%-vol asset is worth about two "
           "hundredths of a Sharpe point. Real, and invisible in a performance table."),
        code(
            "print(f\"own cheapest (GLDM): exSharpe {R['sh_cheap']:.4f}  CAGR {R['cagr_cheap']:+.3f}%\")\n"
            "print(f\"own priciest (GLD) : exSharpe {R['sh_dear']:.4f}  CAGR {R['cagr_dear']:+.3f}%\")\n"
            "print(f\"chase last winner  : exSharpe {R['sh_chase']:.4f}  CAGR {R['cagr_chase']:+.3f}%  \"\n"
            "      f\"({R['n_switches']} switches, 2 bp one-way x NAV)\")\n"
            "print(f\"\\nSharpe gap cheap-dear {R['sh_gap']:+.4f}  -- and NO t is quoted on that gap.\")\n"
            "print(f\"daily return diff cheap-dear: naive t {R['cd_t']:+.2f} [HAC {R['cd_t_hac']:+.2f}]\")\n"
            "print('   the daily view is powerless in both directions; the monthly estimator above')\n"
            "print('   is the evidence, and this race only shows what 26 bp/yr LOOKS like in a')\n"
            "print('   performance table: two hundredths of a Sharpe point. Real, and invisible.')\n"
            "print(f\"chase vs own-cheapest: t {R['chase_t']:+.2f} [HAC {R['chase_t_hac']:+.2f}]  \"\n"
            "      f\"-> the persistence rule LOSES\")\n"
            "print('\\nlag: ranked on data through the last session of the calendar year, FILLED at')\n"
            "print('the next session close, so the new wrapper first earns on the session after')\n"
            "print('that -- never a same-close fill. That is the only decision in the study, and')\n"
            "print('the only place a lag or a cost can apply. No short leg here, so no borrow.')"
        ),
        md("## 8. The four counterweights, all swept"),
        code(
            "print('LIQUIDITY (real tape, close x volume, last 12 months):')\n"
            "for t in %r:\n"
            "    print(f\"  {t:6s} ${R['adv'][t]:6,}m/day   a $100m ticket = {R['adv_days'][t]:5.2f} days of ADV\")\n"
            "print(f\"\\nEXECUTION differential (ASSUMPTION, swept): 5 bp round trip repaid in \"\n"
            "      f\"{R['be5']:.0f} days, 10 bp in {R['be10']:.0f}, a punitive 30 bp in {R['be30']:.0f}\")\n"
            "print(f\"TAX on switching (ASSUMPTION, swept): 20%% on a doubled position takes \"\n"
            "      f\"{R['tax_20_50']:.1f} years;\")\n"
            "print(f\"   23.8%% collectibles rate on a 4x position takes {R['tax_238_300']:.1f} years\")\n"
            "print(f\"LONG/SHORT (ASSUMPTION, swept): gross {R['spread']:.2f} bp/yr becomes \"\n"
            "      f\"{R['ls_gross']:.2f} after costs\")\n"
            "print(f\"   and dies above ~{R['ls_dead_above']:.0f} bp/yr of borrow -> not a trade\")"
            % (ORDER,)
        ),
        md("## 9. Live synthetic control — the machinery is unbiased\n\n"
           "**Synthetic, not the real tape.** Planted ladder: the stack MUST recover it. Null "
           "(every wrapper charging the cohort-average fee while the published sheet still "
           "shows dispersion): it MUST stay quiet."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from which_gold import data, strategy as st\n"
            "pl_p, pl_t = data.synthetic_panel(signal_strength=1.0, seed=961)\n"
            "pl = st.synthetic_detect(pl_p, pl_t)\n"
            "print('SYNTHETIC planted: gap %.1f -> recovered %+.2f bp/yr (t=%+.2f), Spearman %+.3f '\n"
            "      '(p=%.4f), pass-through %+.3f (R2 %.3f)'\n"
            "      % (pl['planted_gap_bpy'], pl['pair_spread_bpy'], pl['pair_t'], pl['spearman'],\n"
            "         pl['p_perm'], pl['pass_through_slope'], pl['pass_through_r2']))\n"
            "nl = [st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=961+s)) for s in range(8)]\n"
            "sp = np.array([n['pair_spread_bpy'] for n in nl])\n"
            "tt = np.array([n['pair_t'] for n in nl])\n"
            "pp = np.array([n['p_perm'] for n in nl])\n"
            "print('SYNTHETIC null x8: spread mean %+.2f bp/yr (sd %.2f), |t|>=2 in %d/8, p<0.05 in %d/8'\n"
            "      % (sp.mean(), sp.std(ddof=1), int((abs(tt) >= 2).sum()), int((pp < 0.05).sum())))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** Cheapest minus priciest is **{R['spread']:+.2f} bp/yr, naive "
           f"*t* = {R['spread_t']:+.2f}** (HAC {R['spread_t_hac']:+.2f}, and inflated — see the "
           f"bandwidth sweep), **iid** bootstrap CI **[{R['ci_lo']:+.1f}, {R['ci_hi']:+.1f}]** "
           f"clear of zero, {R['spread_up']}/{R['n_months']} months and "
           f"{R['pos_years']}/{R['n_years']} calendar years positive, positive in both eras "
           f"({R['era_e']:+.1f} / {R['era_l']:+.1f}, significant in the late one) and "
           f"**{R['stable']:+.1f} at *t* = {R['stable_t']:+.2f} on the fee-stable sub-window**, "
           f"the one cut in which no fee assumption does any work. The fee ranking inverts the "
           f"outcome ranking **perfectly** (Spearman {R['spearman']:.4f}, at its "
           f"{R['p_perm']:.4f} permutation floor) with pass-through "
           f"**{R['pt_slope']:.2f} to {R['pt_slope_blend']:.2f}**, R² **0.99** and ten of ten "
           f"pairwise signs correct; the mechanism is contractual, not discovered. Named limits: "
           f"only the coarse ranking resolves — on the honest *t* one pair clears |*t*| = 2, the "
           f"two 22-bp pairs sit at 1.9 and the four sub-10-bp gaps at |*t*| ≤ 0.6 against a "
           f"{R['floor_bpy']:.1f} bp/yr floor; the early era is not significant alone; the fee "
           f"sheet is an ASSUMPTION carrying hindsight and the headline pair is a selection off "
           f"it; the cohort is five survivors.\n"
           f"- **Tradability — Investable.** Not because the tape pins the number down — the "
           f"honest interval is {R['ci_lo']:.0f} to {R['ci_hi']:.0f} bp — but because the act is "
           f"a **purchase decision with no forecast, no timing, no turnover and no capacity "
           f"constraint** on the wrapper that wins, and because even at the bottom of that "
           f"interval the cheap wrapper is not the worse buy — a dominance argument, not a "
           f"forecast (GLDM turns ${R['adv']['GLDM']}m a day and "
           f"repays a 30 bp execution penalty in {R['be30']:.0f} days), worth "
           f"**{R['cost10']:.1f}% of terminal wealth over ten years**. It is *not* a reason to "
           f"sell an appreciated position ({R['tax_20_50']:.0f}–{R['tax_238_300']:.0f} years to "
           f"repay the tax), *not* a long/short (dead above ~{R['ls_dead_above']:.0f} bp of "
           f"borrow), and *not* an argument for the fee-cheapest name at size (BAR: "
           f"${R['adv']['BAR']}m/day, {R['adv_days']['BAR']:.1f} days of ADV for a $100m "
           f"ticket)."),
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
