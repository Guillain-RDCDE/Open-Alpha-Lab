"""Generate the two narrative notebooks for Study 945 (The Hidden Financing).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Real-tape headline numbers are quoted from the frozen ``R`` dict below, which mirrors
``docs/results.md`` — no cell recomputes them from the network. The live cells are the
offline synthetic control (fast, deterministic, network-free) and one optional real-cache
chart that says so loudly and never falls back to synthetic data.
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


# Frozen real-tape headline — mirror of docs/results.md. SSO (2x) and UPRO (3x) vs SPY
# total return, BIL cash leg, ^IRX rate, common window 2009-06-26 -> 2026-06-30.
R = dict(
    start="2009-06-26", end="2026-06-30", n_days=4276, fp="1c447f91bfae",
    irx_mean=1.396, bil_realised=1.267,
    sso_beta=1.9972, sso_beta_t=-0.27, sso_r2=0.99748,
    sso_alpha=-2.774, sso_alpha_se=0.208, sso_alpha_t=-13.31,
    sso_drag=2.962, sso_f=2.072, sso_spread=0.677,
    sso_allin=2.962, sso_allin_spread=1.567,
    upro_beta=2.9894, upro_beta_t=-0.63, upro_r2=0.99653,
    upro_alpha=-4.787, upro_alpha_se=0.375, upro_alpha_t=-12.75,
    upro_drag=5.070, upro_f=2.080, upro_spread=0.684,
    upro_allin=2.535, upro_allin_spread=1.139,
    boot_sso=0.721, boot_sso_lo=0.390, boot_sso_hi=1.079,
    boot_upro=0.768, boot_upro_lo=0.432, boot_upro_hi=1.114,
    gspc_sso_drag=-0.888, gspc_sso_f=-1.778, gspc_upro_drag=-0.694, gspc_upro_f=-0.802,
    roll_sso_mean=0.545, roll_sso_sd=0.437, roll_sso_slope=1.058, roll_sso_int=0.466,
    roll_sso_corr=0.975,
    roll_upro_mean=0.603, roll_upro_sd=0.405, roll_upro_slope=1.061, roll_upro_int=0.520,
    roll_upro_corr=0.979,
    # HAC t on the SPREAD (the claim), not on the intercept (which the fee alone makes
    # non-zero). The all-in t needs no expense-ratio assumption at all.
    sso_spread_se=0.208, sso_spread_t=3.25, sso_allin_t=7.52,
    upro_spread_se=0.188, upro_spread_t=3.65, upro_allin_t=6.07,
    irx_bey=1.420,
    era_e_spread=0.374, era_l_spread=0.951, era_e_t=1.74, era_l_t=3.33,
    zirp_spread=0.658, hiked_spread=0.635, zirp_t=2.69, hiked_t=2.08,
    upro_era_e_spread=0.380, upro_era_l_spread=0.962,
    upro_era_e_t=1.77, upro_era_l_t=3.89,
    upro_zirp_spread=0.631, upro_hiked_spread=0.711,
    upro_zirp_t=2.83, upro_hiked_t=2.80,
    be_sso_early=1.167, be_sso_late=1.607, be_sso_zirp=1.169, be_sso_hiked=1.748,
    be_upro_early=0.811, be_upro_late=1.255, be_upro_zirp=0.827, be_upro_hiked=1.364,
    er_lo_spread=0.817, er_hi_spread=0.517,
    sharpe_spy=0.836, sharpe_sso=0.792, sharpe_upro=0.792,
    race_sso_075=-0.64, race_sso_075_t=-4.56,
    race_sso_150=0.11, race_sso_150_t=0.78,
    race_sso_400=2.61, race_sso_400_t=18.59,
    race_sso_600=4.61, race_sso_600_t=32.84,
    race_upro_075=-0.57, race_upro_075_t=-2.25,
    race_upro_400=5.93, race_upro_400_t=23.36,
    race_upro_600=9.93, race_upro_600_t=39.12,
    be_sso_gross=1.427, be_sso=1.390, be_sso_5bp=1.245,
    be_upro_gross=1.090, be_upro=1.035, be_upro_5bp=0.817,
    turn_sso=1.443, turn_upro=4.330, turn_cost_sso=0.04, turn_cost_upro=0.11,
    syn_planted=0.750, syn_2x=0.683, syn_3x=0.813, syn_err_2x=-0.067, syn_err_3x=0.063,
)

HEADER = f"""# Study 945 — The Hidden Financing 💳

**What interest rate are you really paying inside a leveraged ETF?**

Buy $10,000 of a 2x S&P fund and you control $20,000 of index. Somebody lent you the other
$10,000. The fact sheet quotes an expense ratio; it does not quote the *interest rate* on
that loan, which arrives silently inside a swap spread. But it is recoverable, because the
fund's arithmetic is rigid:

$$r_{{fund}} = L \\cdot r_{{index}} - (L-1)\\cdot \\frac{{f}}{{252}} - \\frac{{ER}}{{252}} + \\varepsilon$$

Regress the fund's daily total return on the benchmark's: the **slope** is the realised
leverage, the **intercept** is the whole daily drag. Annualise it, strip the published
expense ratio, divide by the `L−1` dollars actually borrowed — what is left is *f*, the
implied financing rate.

We do it on **SSO** (2x) and **UPRO** (3x) against **SPY** total return, {R['start']} →
{R['end']} ({R['n_days']:,} days), and race the answer against **^IRX** (the 13-week
T-bill) and against what a margin desk would charge.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
live cells run the offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The loan nobody quotes you\n\n"
           "A leveraged ETF is two things bolted together: an index position and a margin "
           "loan. You are told the price of the wrapper (the expense ratio). You are not "
           "told the price of the loan. So we read it off the tape instead — the fund's "
           "return has to be the index return times the leverage, *minus* the fee, *minus* "
           "the interest. Everything except the interest is known, so the interest falls out."),
        code(
            "R = " + repr(R) + "\n"
            "print('Mean 13-week T-bill rate over the window : %.3f%%' % R['irx_mean'])\n"
            "print()\n"
            "rows = (('SSO  (2x)', R['sso_f'], R['sso_spread'], R['sso_allin'], R['sso_allin_spread']),\n"
            "        ('UPRO (3x)', R['upro_f'], R['upro_spread'], R['upro_allin'], R['upro_allin_spread']))\n"
            "for tag, f, sp, allin, allsp in rows:\n"
            "    print('%s implied borrowing rate %5.3f%%  = T-bills %+.3f pp' % (tag, f, sp))\n"
            "    print('%s all-in, incl. the fee   %5.3f%%  = T-bills %+.3f pp'\n"
            "          % (' ' * 9, allin, allsp))"
        ),
        md(f"## 2. The two funds agree to within one basis point\n\n"
           f"SSO borrows one dollar per dollar of your money; UPRO borrows two. Solve each "
           f"one separately and they imply the *same* borrowing rate: **{R['sso_f']:.3f}%** "
           f"and **{R['upro_f']:.3f}%** — eight tenths of a basis point apart. Against a "
           f"mean T-bill rate of {R['irx_mean']:.3f}%, that is a mark-up of about "
           f"**+{R['sso_spread']:.2f} pp** for borrowing money, with a HAC *t* of "
           f"**{R['sso_spread_t']:.2f}** and **{R['upro_spread_t']:.2f}**.\n\n"
           f"Two cautions, because they matter more than the agreement does:\n\n"
           f"- The two funds are **both ProShares**, sharing an issuer and a swap desk. "
           f"Agreeing tells you the arithmetic is not broken; it is not two independent "
           f"witnesses to the level.\n"
           f"- What is left after stripping the fee is the interest **plus** every other "
           f"friction inside the wrapper — swap spreads, the cost of the daily reset, "
           f"tracking loss. A return regression cannot separate them, so read "
           f"{R['sso_f']:.2f}% as an **upper bound** on the interest rate, not the rate.\n\n"
           f"> 🔬 *For the quants:* the intercepts are −{abs(R['sso_alpha']):.3f}%/yr "
           f"(*t* = {R['sso_alpha_t']:.2f}) and −{abs(R['upro_alpha']):.3f}%/yr "
           f"(*t* = {R['upro_alpha_t']:.2f}) — but those *t*'s only say the drag is "
           f"non-zero, which the fee alone guarantees. The claim is the spread, at "
           f"*t* = {R['sso_spread_t']:.2f} / {R['upro_spread_t']:.2f}. Realised betas "
           f"{R['sso_beta']:.4f} and {R['upro_beta']:.4f}, neither distinguishable from "
           f"its stated leverage, so the intercept is a level term and not a bad slope."),
        md(f"## 3. The 3x fund is the *cheaper* loan\n\n"
           f"Counter-intuitive, and it falls straight out of where the fee lands. The "
           f"expense ratio is charged on your **whole** stake; the loan is only part of it. "
           f"At 2x you pay the fee on $1 to borrow $1. At 3x you pay (almost) the same fee "
           f"on $1 to borrow **$2** — so the fee is spread over twice the borrowing.\n\n"
           f"| | borrowed per $1 | drag on your money | all-in cost per borrowed dollar |\n"
           f"|---|--:|--:|--:|\n"
           f"| **SSO** (2x) | $1.00 | {R['sso_drag']:.2f}%/yr | **{R['sso_allin']:.2f}%** "
           f"(T-bills +{R['sso_allin_spread']:.2f}) |\n"
           f"| **UPRO** (3x) | $2.00 | {R['upro_drag']:.2f}%/yr | **{R['upro_allin']:.2f}%** "
           f"(T-bills +{R['upro_allin_spread']:.2f}) |\n\n"
           f"None of which says 3x is *safer* — it is far more violent, and this desk has "
           f"said so twice already (studies 61 and 100). It says only that the interest "
           f"you pay per borrowed dollar is lower."),
        md(f"## 4. When the Fed moves, your loan moves — one for one\n\n"
           f"Re-estimate the rate on a rolling one-year window through 2009-2026 and the "
           f"implied borrowing rate tracks the T-bill rate almost perfectly (correlation "
           f"**{R['roll_sso_corr']:.3f}**), with a slope of **{R['roll_sso_slope']:.2f}** and "
           f"a roughly constant mark-up on top. It was 0.11% in 2013 and 6.06% in 2024 — "
           f"the whole rate cycle, passed through.\n\n"
           f"So the wrapper is not a fixed-rate loan you locked in. It is a floating-rate "
           f"loan, and whatever the Fed does to short rates lands on you within the year."),
        md(f"## 5. So — wrapper, or your own broker?\n\n"
           f"That is the whole practical question, and the tape answers it with a single "
           f"number: the **break-even margin rate**. Hold 2x SPY yourself on margin, reset "
           f"daily exactly as the fund does, pay one basis point each time you trade. You "
           f"tie the fund when your broker charges **T-bills + {R['be_sso']:.2f}%**; at 3x "
           f"the line is **T-bills + {R['be_upro']:.2f}%**.\n\n"
           f"| Your broker charges | 2x: wrapper minus DIY | 3x: wrapper minus DIY |\n"
           f"|---|--:|--:|\n"
           f"| bills + 0.75% (prime-broker tier) | **{R['race_sso_075']:+.2f}%/yr** | "
           f"**{R['race_upro_075']:+.2f}%/yr** |\n"
           f"| bills + 1.50% (low-cost retail) | {R['race_sso_150']:+.2f}%/yr | — |\n"
           f"| bills + 4.00% (mainstream broker) | **{R['race_sso_400']:+.2f}%/yr** | "
           f"**{R['race_upro_400']:+.2f}%/yr** |\n"
           f"| bills + 6.00% (full-service retail) | **{R['race_sso_600']:+.2f}%/yr** | "
           f"**{R['race_upro_600']:+.2f}%/yr** |\n\n"
           f"Almost every ordinary brokerage account is on the bottom two rows, where the "
           f"wrapper wins by several percent a year. If you are on the top row you are "
           f"already an institution and you knew that.\n\n"
           f"Two honest limits on that table. **You may not be allowed to do it yourself:** "
           f"Reg T caps a US retail margin account at 2x to begin with, so the whole 3x "
           f"column is a price comparison rather than a choice you can make. And the "
           f"break-even is a full-sample average that moves when you cut the sample — "
           f"^IRX +{R['be_sso_early']:.2f}% to +{R['be_sso_late']:.2f}% at 2x across the two "
           f"halves. It never climbs anywhere near the 4-6% a retail desk charges, which is "
           f"why the answer holds; but it is a range, not a constant.\n\n"
           f"> 🔬 *For the quants:* the broker rates are **labelled assumptions** from "
           f"public rate cards; the break-even itself contains no broker assumption at all. "
           f"Both arms are raced excess-of-cash against BIL's realised total return, with "
           f"one execution lag and the daily reset's turnover charged one-way against NAV. "
           f"No margin call is modelled — that flatters the DIY arm, so the break-even is a "
           f"conservative line for the wrapper."),
        md("## 6. Live check — does the arithmetic actually recover a known rate? (offline)\n\n"
           "We build synthetic wrappers whose financing rate we *chose*: benchmark + 75 bp, "
           "plus a 0.90% fee, over a 24-year tape with a rate cycle in it. Then we run the "
           "exact estimator used above and see whether it hands the 75 bp back. And we run "
           "it again on a null world where the wrappers borrow at exactly the benchmark "
           "rate and charge nothing — where it must return zero."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from lev_financing import data, strategy as st\n"
            "print('SYNTHETIC (offline) — not the real tape')\n"
            "for ss, tag in ((1.0, 'planted 75 bp'), (0.0, 'null      0 bp')):\n"
            "    for L in (2, 3):\n"
            "        got = [st.synthetic_detect(*data.synthetic_panel(signal_strength=ss, seed=945+s),\n"
            "                                   leverage=L)['spread_over_rate_pct'] for s in range(6)]\n"
            "        print('  %s  %dx -> recovered spread %+.3f pp (6 seeds)' % (tag, L, np.mean(got)))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The charge is measured, not guessed. The mark-up over "
           f"T-bills is **+{R['sso_spread']:.2f} pp** at a HAC *t* of "
           f"**{R['sso_spread_t']:.2f}** (SSO) and **{R['upro_spread_t']:.2f}** (UPRO), and "
           f"the all-in charge — which needs no guess about the fee at all — is "
           f"**+{R['sso_allin_spread']:.2f} pp** at *t* = **{R['sso_allin_t']:.1f}**. "
           f"Robust to any plausible error in the assumed expense ratio, and positive in "
           f"both rate worlds.\n"
           f"- **Three caveats that ride with the badge.** It is an **upper bound** on "
           f"interest (swap and reset frictions are in there too). The **first half of the "
           f"sample is weak** — +{R['era_e_spread']:.2f} pp at *t* = {R['era_e_t']:.2f}, "
           f"below the bar — so what is solid is the post-2018 level near "
           f"+{R['era_l_spread']:.2f}. And these are the funds that **survived**; the ones "
           f"that closed are not in the average, so this is a floor for the class.\n"
           f"- **Tradability — Investable.** Not an edge — a cost decision, and a big one. "
           f"All-in you pay bills + {R['sso_allin_spread']:.2f} pp at 2x and "
           f"+{R['upro_allin_spread']:.2f} pp at 3x. Against a mainstream retail margin desk "
           f"that is worth **+{R['race_sso_400']:.1f}% to +{R['race_upro_600']:.1f}% a year** "
           f"in your favour — a gap wide enough that the wobble in the break-even "
           f"(^IRX +{R['be_sso_early']:.2f}% to +{R['be_sso_late']:.2f}% across the two "
           f"halves) never threatens the answer. Against a prime broker it is worth "
           f"**{R['race_sso_075']:.2f}%** against you — but that tier, and the 3x "
           f"do-it-yourself arm Reg T does not allow a retail account, are not on offer to "
           f"the reader this rule is written for. **If you are levering 2x anyway, the "
           f"wrapper is the cheaper loan unless your broker lends under about bills + 1.2%.**\n"
           f"- **What this does not say.** Nothing here is an argument for *using* leverage. "
           f"It prices the loan; the shape of the ride — the volatility drag, the −60% and "
           f"−77% drawdowns these two funds actually took — is studies 61, 100 and 944."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 945 — The Hidden Financing — the teardown\n\n"
           "The HAC regression that inverts the constant-leverage identity, the price-index "
           "trap, the block-bootstrap CI, the rolling pass-through through the rate cycle, "
           "the era and rate-regime cuts, the expense-ratio sweep, the costed break-even "
           "race against a self-financed margin replication, and the live synthetic control. "
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`).\n\n"
           "> 💡 *In plain words:* a leveraged ETF is an index position plus a margin loan. "
           "This notebook works out the interest rate on the loan." % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The identity and its inversion\n\n"
           "$$r_{fund,t} = \\beta\\, r_{bench,t} + \\alpha + \\varepsilon_t, \\qquad "
           "\\beta \\to L, \\quad \\alpha \\to -\\frac{(L-1)f + ER}{252}$$\n\n"
           "Two accounting details decide whether the answer is a rate or a nonsense:\n\n"
           "1. **Basis.** The fund tracks the *index*; SPY runs `SPY_ER = 0.0945%`/yr below "
           "it, so the drag on an index basis is `−α·252 + β·SPY_ER`.\n"
           "2. **Total return on both sides.** The wrappers distribute part of their "
           "collateral income; `auto_adjust=True` puts it back, so the measured drag is the "
           "*net* cost rather than an artefact of what was paid out.\n\n"
           "One execution lag and only one, and it is not in this regression: the estimation "
           "half is a **measurement**, so fund and benchmark returns are contemporaneous by "
           "construction. The lag lives in the replication race in §7."),
        code(
            "print('OLS, Newey-West SEs, drag on an index basis, %s -> %s (n=%d)'\n"
            "      % (R['start'], R['end'], R['n_days']))\n"
            "hdr = '{:<6} {:>8} {:>7} {:>10} {:>7} {:>8} {:>9} {:>9}'\n"
            "print(hdr.format('fund', 'beta', 't vs L', 'alpha%/yr', 'se', 't',\n"
            "                 'drag%/yr', 'f_impl%'))\n"
            "for tag, k in (('SSO', 'sso'), ('UPRO', 'upro')):\n"
            "    print(hdr.format(tag,\n"
            "                     '{:.4f}'.format(R[k + '_beta']),\n"
            "                     '{:+.2f}'.format(R[k + '_beta_t']),\n"
            "                     '{:+.3f}'.format(R[k + '_alpha']),\n"
            "                     '{:.3f}'.format(R[k + '_alpha_se']),\n"
            "                     '{:+.2f}'.format(R[k + '_alpha_t']),\n"
            "                     '{:.3f}'.format(R[k + '_drag']),\n"
            "                     '{:.3f}'.format(R[k + '_f'])))\n"
            "print()\n"
            "print('mean ^IRX %.3f%%  ->  spread SSO %+.3f pp / UPRO %+.3f pp'\n"
            "      % (R['irx_mean'], R['sso_spread'], R['upro_spread']))\n"
            "print('all-in per borrowed dollar: SSO %.3f%% (^IRX %+.3f) / UPRO %.3f%% (^IRX %+.3f)'\n"
            "      % (R['sso_allin'], R['sso_allin_spread'], R['upro_allin'],\n"
            "         R['upro_allin_spread']))\n"
            "print()\n"
            "print('THE BAR — HAC t on the CLAIMS, not on the intercept:')\n"
            "print('  spread over ^IRX  : SSO %+.2f / UPRO %+.2f   (leans on the assumed ER)'\n"
            "      % (R['sso_spread_t'], R['upro_spread_t']))\n"
            "print('  all-in over ^IRX  : SSO %+.2f / UPRO %+.2f   (no ER assumption at all)'\n"
            "      % (R['sso_allin_t'], R['upro_allin_t']))\n"
            "print('  the intercept t of %.2f only says the drag is non-zero, which the'\n"
            "      % R['sso_alpha_t'])\n"
            "print('  0.89% expense ratio guarantees before a cent of interest is charged.')"
        ),
        md(f"Realised betas of {R['sso_beta']:.4f} (*t* vs 2 = {R['sso_beta_t']:+.2f}) and "
           f"{R['upro_beta']:.4f} (*t* vs 3 = {R['upro_beta_t']:+.2f}) — the wrappers do "
           f"exactly what they say on the daily horizon, so the intercept is not soaking up "
           f"a slope error. R² of {R['sso_r2']:.5f} / {R['upro_r2']:.5f}.\n\n"
           f"**Which *t* carries the claim.** The intercept's {R['sso_alpha_t']:.2f} tests "
           f"only that the drag is non-zero — guaranteed by a 0.89% expense ratio before any "
           f"interest is charged — so it is not the inference bar and is never quoted as "
           f"one. The claim is the **spread**: *t* = {R['sso_spread_t']:.2f} / "
           f"{R['upro_spread_t']:.2f} (same intercept SE, rescaled by the *L*−1 borrowed "
           f"dollars). The **all-in** spread, which carries no expense-ratio assumption, "
           f"runs *t* = {R['sso_allin_t']:.2f} / {R['upro_allin_t']:.2f}.\n\n"
           f"**What the residual contains.** Fee stripped, what remains is financing *plus* "
           f"swap spreads, daily-reset slippage and residual tracking loss — inseparable in "
           f"a return regression. **{R['sso_f']:.3f}% is an upper bound on the borrowing "
           f"rate**, not the borrowing rate. The all-in cost per borrowed dollar needs no "
           f"such caveat.\n\n"
           f"The cross-wrapper agreement (**{R['sso_f']:.3f}%** vs **{R['upro_f']:.3f}%**, "
           f"0.8 bp apart) is a genuine check — nothing in the estimator forces it — but not "
           f"an independent one: both are ProShares funds on a shared swap desk. It rules "
           f"out a mis-specified estimator; it does not confirm the level twice."),
        md(f"## 2. The price-index trap\n\n"
           f"\"The S&P 500\" as quoted (^GSPC) is a **price** index. Regress on it and the "
           f"missing dividend yield, multiplied by *L*, swamps the drag:\n\n"
           f"| Benchmark | SSO drag | SSO implied *f* | UPRO drag | UPRO implied *f* |\n"
           f"|---|--:|--:|--:|--:|\n"
           f"| SPY **total return** (correct) | {R['sso_drag']:+.3f}% | {R['sso_f']:.3f}% | "
           f"{R['upro_drag']:+.3f}% | {R['upro_f']:.3f}% |\n"
           f"| ^GSPC **price** index (wrong) | {R['gspc_sso_drag']:+.3f}% | "
           f"**{R['gspc_sso_f']:+.3f}%** | {R['gspc_upro_drag']:+.3f}% | "
           f"**{R['gspc_upro_f']:+.3f}%** |\n\n"
           f"A wrapper borrowing at −1.8% would be a money machine. It isn't; the benchmark "
           f"was wrong. This is the single most common way published estimates of leveraged-"
           f"ETF financing go wrong.\n\n"
           f"> 💡 *In plain words:* the index you see on television leaves dividends out. "
           f"The fund doesn't. Compare like with like or the answer inverts."),
        md(f"## 3. Block-bootstrap CI on the spread\n\n"
           f"Resample the **slope-pinned** daily drag series `L·r_SPY − r_fund` in 21-day "
           f"circular blocks (preserving tracking-error clustering) and push each draw "
           f"through the same arithmetic. The pinned point sits ~4 bp above the free-slope "
           f"one — precisely the disagreement you expect when β is 0.003 from *L*.\n\n"
           f"| Fund | Spread | 95% CI | share < 0 |\n|---|--:|--:|--:|\n"
           f"| SSO | {R['boot_sso']:+.3f} pp | [{R['boot_sso_lo']:+.3f}, "
           f"{R['boot_sso_hi']:+.3f}] | 0.0% |\n"
           f"| UPRO | {R['boot_upro']:+.3f} pp | [{R['boot_upro_lo']:+.3f}, "
           f"{R['boot_upro_hi']:+.3f}] | 0.0% |"),
        md(f"## 4. Rolling pass-through through the rate cycle\n\n"
           f"Rolling 252-day OLS, each estimate stamped on its window's last day (causal by "
           f"construction), with the companion ^IRX figure the trailing mean over the "
           f"identical window. Then regress the rolling implied *f* on the rolling ^IRX:\n\n"
           f"| Fund | spread mean | sd | pass-through slope | intercept | corr |\n"
           f"|---|--:|--:|--:|--:|--:|\n"
           f"| SSO | {R['roll_sso_mean']:+.3f} | {R['roll_sso_sd']:.3f} | "
           f"**{R['roll_sso_slope']:.3f}** | {R['roll_sso_int']:+.3f} | "
           f"{R['roll_sso_corr']:.3f} |\n"
           f"| UPRO | {R['roll_upro_mean']:+.3f} | {R['roll_upro_sd']:.3f} | "
           f"**{R['roll_upro_slope']:.3f}** | {R['roll_upro_int']:+.3f} | "
           f"{R['roll_upro_corr']:.3f} |\n\n"
           f"Slope ≈ 1 with a +0.5 pp intercept is the signature of a **pure pass-through "
           f"with a constant mark-up**. Caveat stated plainly: overlapping 252-day windows "
           f"leave these residuals massively autocorrelated, so the slope is *descriptive*; "
           f"the inference bar is carried by the full-sample regression in §1."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "%matplotlib inline\n"
            "import matplotlib.pyplot as plt\n"
            "from lev_financing import data, strategy as st\n"
            "\n"
            "# REAL TAPE if the shared cache is present; otherwise say so and plot NOTHING.\n"
            "# There is no synthetic fallback under this banner.\n"
            "if not data.have_real():\n"
            "    print('shared studies/_cache absent -> chart skipped '\n"
            "          '(run lev_financing.data.fetch() once; no synthetic stand-in is drawn)')\n"
            "else:\n"
            "    px = data.load_prices().dropna()\n"
            "    rets = px[['SSO', 'UPRO', 'SPY']].pct_change().dropna()\n"
            "    rate = px['IRX'].reindex(rets.index)\n"
            "    fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "    ax.plot(rate.index, rate, lw=1.0, color='0.45', label='^IRX (13-week bill)')\n"
            "    for fund, colour in (('SSO', '#1f77b4'), ('UPRO', '#d62728')):\n"
            "        roll = st.rolling_financing(rets[fund], rets['SPY'], rate,\n"
            "                                    data.FUNDS[fund], data.EXPENSE_RATIO[fund],\n"
            "                                    bench_fee=data.SPY_EXPENSE_RATIO, window=252)\n"
            "        ax.plot(roll.index, roll['implied_financing'], lw=1.3, color=colour,\n"
            "                label='%s implied financing (252d)' % fund)\n"
            "    ax.set_title('REAL TAPE — implied financing rate vs the 13-week bill')\n"
            "    ax.set_ylabel('% per annum'); ax.legend(loc='upper left', fontsize=8)\n"
            "    ax.grid(alpha=0.25)\n"
            "    fig.tight_layout(); plt.show()\n"
            "    print('fingerprint', data.fingerprint(px), '| as-of', data.AS_OF)"
        ),
        md(f"## 5. Two cuts — calendar era and rate regime\n\n"
           f"The 2018-01-01 split is the sample **midpoint** (2,142 days against 2,134), not "
           f"a date anyone chose. Every *t* below is the **spread's** own HAC *t*.\n\n"
           f"| Cut | SSO spread (*t*) | UPRO spread (*t*) |\n|---|--:|--:|\n"
           f"| 2009–2017 | {R['era_e_spread']:+.3f} (**{R['era_e_t']:+.2f}** ⚠) | "
           f"{R['upro_era_e_spread']:+.3f} (**{R['upro_era_e_t']:+.2f}** ⚠) |\n"
           f"| 2018–2026 | **{R['era_l_spread']:+.3f}** ({R['era_l_t']:+.2f}) | "
           f"**{R['upro_era_l_spread']:+.3f}** ({R['upro_era_l_t']:+.2f}) |\n"
           f"| ^IRX < 1% | **{R['zirp_spread']:+.3f}** ({R['zirp_t']:+.2f}) | "
           f"**{R['upro_zirp_spread']:+.3f}** ({R['upro_zirp_t']:+.2f}) |\n"
           f"| ^IRX ≥ 1% | **{R['hiked_spread']:+.3f}** ({R['hiked_t']:+.2f}) | "
           f"**{R['upro_hiked_spread']:+.3f}** ({R['upro_hiked_t']:+.2f}) |\n\n"
           f"The **rate-regime** cut is the one that passes cleanly: a 60 bp mark-up is most "
           f"of the cost at the zero bound and a garnish at 5%, and both halves clear "
           f"|*t*| = 2 at near-identical levels ({R['zirp_spread']:.3f} vs "
           f"{R['hiked_spread']:.3f} for SSO) — a structural charge, not a zero-bound "
           f"artefact.\n\n"
           f"⚠ **The calendar cut is where this study is weakest, and it is not buried.** "
           f"Pre-2018 the spread is {R['era_e_spread']:+.3f} / "
           f"{R['upro_era_e_spread']:+.3f} at *t* = {R['era_e_t']:.2f} / "
           f"{R['upro_era_e_t']:.2f} — **below the bar**: that half of the tape, alone, "
           f"cannot reject \"the wrappers borrowed at bills\". After 2018 the mark-up roughly "
           f"doubles ({R['era_l_spread']:.2f} / {R['upro_era_l_spread']:.2f}, *t* = "
           f"{R['era_l_t']:.2f} / {R['upro_era_l_t']:.2f}). Read it as: the **post-2018** "
           f"level near +0.95 is what the tape establishes; the full-sample +0.68 is an "
           f"average over a first half that is a weak estimate of a smaller number, and it "
           f"is not a forward constant."),
        md(f"## 6. The one non-tape input, swept\n\n"
           f"The **drag is measured**; only its split between fee and financing depends on the "
           f"prospectus expense ratio. A 10 bp error moves implied *f* by 10 bp/(*L*−1), so the "
           f"3x estimate is twice as robust to it. Across the whole ±15 bp band the spread runs "
           f"from **+{R['er_lo_spread']:.3f}** (ER 0.75%) to **+{R['er_hi_spread']:.3f}** "
           f"(ER 1.05%) for SSO — positive throughout. **There is no plausible expense ratio at "
           f"which these wrappers borrow at T-bills.**\n\n"
           f"> 💡 *In plain words:* even if the published fee is wrong by a fifth, the "
           f"conclusion doesn't move."),
        md(f"## 7. The race — wrapper vs self-financed daily-reset replication\n\n"
           f"The DIY arm holds *L*×SPY on margin at ^IRX + spread and resets to *L* every "
           f"close, the same mechanic the fund runs. **One execution lag:** the position "
           f"earning day *t*'s return was set at the close of *t−1*, and the rate charged over "
           f"day *t* is the one quoted at the close of *t−1*. Reset turnover is "
           f"*L*(*L*−1)|r| of NAV — {R['turn_sso']:.2f}%/day at 2x, {R['turn_upro']:.2f}%/day "
           f"at 3x — charged one-way **against NAV** ({R['turn_cost_sso']:.2f}%/yr and "
           f"{R['turn_cost_upro']:.2f}%/yr at 1 bp, i.e. the reset is *not* the expensive "
           f"part). Both arms excess-of-cash against BIL's realised total return. **No short "
           f"leg anywhere, so no borrow fee** — the only borrowing is the long leverage being "
           f"priced.\n\n"
           f"**Eligibility, not just price.** Reg T caps a US retail margin account's "
           f"*initial* leverage at 2x, so the **3x DIY arm does not exist** in an ordinary "
           f"account — it needs portfolio margin, futures or a prime broker. Every UPRO row "
           f"below, and the +0.75% row at either leverage, is a *price* comparison rather "
           f"than a choice a retail holder can make. No margin call is modelled either; the "
           f"wrapper's loss is capped at NAV and the margin account's is not. Both omissions "
           f"flatter the DIY arm, so the break-even is **conservative** for the wrapper.\n\n"
           f"**The *t* column is not evidence of an edge.** Both arms hold the same *L* × the "
           f"same benchmark, so the difference is a near-deterministic cost gap plus tracking "
           f"noise; its *t* grows mechanically with the assumed margin spread and with sample "
           f"length. *t* = {R['race_upro_600_t']:.0f} means \"the arithmetic is stable\", not "
           f"\"{R['race_upro_600_t']:.0f} sigma of alpha\".\n\n"
           f"| Margin spread over ^IRX | SSO: fund − DIY (*t*) | UPRO: fund − DIY (*t*) |\n"
           f"|---|--:|--:|\n"
           f"| +0.75% | {R['race_sso_075']:+.2f}%/yr ({R['race_sso_075_t']:+.2f}) | "
           f"{R['race_upro_075']:+.2f}%/yr ({R['race_upro_075_t']:+.2f}) |\n"
           f"| +1.50% | {R['race_sso_150']:+.2f}%/yr ({R['race_sso_150_t']:+.2f}) | — |\n"
           f"| +4.00% | **{R['race_sso_400']:+.2f}%/yr** ({R['race_sso_400_t']:+.2f}) | "
           f"**{R['race_upro_400']:+.2f}%/yr** ({R['race_upro_400_t']:+.2f}) |\n"
           f"| +6.00% | **{R['race_sso_600']:+.2f}%/yr** ({R['race_sso_600_t']:+.2f}) | "
           f"**{R['race_upro_600']:+.2f}%/yr** ({R['race_upro_600_t']:+.2f}) |\n\n"
           f"Excess Sharpes: SPY **{R['sharpe_spy']:.3f}**, SSO {R['sharpe_sso']:.3f}, UPRO "
           f"{R['sharpe_upro']:.3f}. Leverage is near Sharpe-neutral before costs; the ~0.045 "
           f"both wrappers give up against SPY *is* the fee plus the financing.\n\n"
           f"**Break-even margin spread** — gross and net, and the answer that contains no "
           f"broker assumption at all:\n\n"
           f"| One-way cost | SSO | UPRO |\n|---|--:|--:|\n"
           f"| 0 bp (gross) | ^IRX +{R['be_sso_gross']:.3f}% | ^IRX +{R['be_upro_gross']:.3f}% |\n"
           f"| 1 bp (net, headline) | **^IRX +{R['be_sso']:.3f}%** | "
           f"**^IRX +{R['be_upro']:.3f}%** |\n"
           f"| 5 bp | ^IRX +{R['be_sso_5bp']:.3f}% | ^IRX +{R['be_upro_5bp']:.3f}% |\n\n"
           f"**And it is an in-sample average that moves when you cut the sample** — quoted "
           f"as a range, never as a constant (1 bp one-way):\n\n"
           f"| Cut | SSO | UPRO |\n|---|--:|--:|\n"
           f"| 2009–2017 | ^IRX +{R['be_sso_early']:.3f}% | ^IRX +{R['be_upro_early']:.3f}% |\n"
           f"| 2018–2026 | ^IRX +{R['be_sso_late']:.3f}% | ^IRX +{R['be_upro_late']:.3f}% |\n"
           f"| ^IRX < 1% | ^IRX +{R['be_sso_zirp']:.3f}% | ^IRX +{R['be_upro_zirp']:.3f}% |\n"
           f"| ^IRX ≥ 1% | ^IRX +{R['be_sso_hiked']:.3f}% | ^IRX +{R['be_upro_hiked']:.3f}% "
           f"|\n\n"
           f"The level swings by ±0.3 pp. What does not move is the decision at realistic "
           f"retail rates: the break-even never exceeds ^IRX +{R['be_sso_hiked']:.2f}% in any "
           f"cut, against retail margin at ^IRX + 4 to 6% — safe by a factor of three. The "
           f"conclusion is robust; the third decimal is not."),
        md("## 8. Live synthetic control (offline — never supports the stamp)\n\n"
           "Synthetic wrappers assembled from a *known* financing rate over 24 years with a "
           "planted 0.15% → 5.00% rate step and 5 bp/day tracking noise. Because the planted "
           "and null worlds share a seed — hence the same index path and the same noise — the "
           "*difference* between the two recovered spreads is the planted effect with zero "
           "residual sampling error, which is what the test-suite asserts as an equality."),
        code(
            "import numpy as np\n"
            "from lev_financing import data, strategy as st\n"
            "print('SYNTHETIC (offline) — not the real tape\\n')\n"
            "for L in (2, 3):\n"
            "    got = {}\n"
            "    for ss in (1.0, 0.0):\n"
            "        vals = [st.synthetic_detect(*data.synthetic_panel(signal_strength=ss, seed=945+s),\n"
            "                                    leverage=L)['spread_over_rate_pct'] for s in range(6)]\n"
            "        got[ss] = np.array(vals)\n"
            "    print('%dx  planted 75 bp -> %+.3f pp (6-seed mean, sd %.3f)'\n"
            "          % (L, got[1.0].mean(), got[1.0].std()))\n"
            "    print('    null    0 bp -> %+.3f pp' % got[0.0].mean())\n"
            "    print('    planted minus null = %+.4f pp (exactly the planted effect)\\n'\n"
            "          % (got[1.0] - got[0.0]).mean())\n"
            "p1, t1 = data.synthetic_panel(signal_strength=1.0, seed=945)\n"
            "roll = st.rolling_financing(p1['fund_3'].pct_change().dropna(),\n"
            "                            p1['index'].pct_change().dropna(), p1['rate'], 3,\n"
            "                            t1['expense_ratio_pct'], window=252)\n"
            "print('rolling estimator on the planted rate step: %.2f%% -> %.2f%% (planted %.2f -> %.2f)'\n"
            "      % (roll['implied_financing'].iloc[:200].mean(),\n"
            "         roll['implied_financing'].iloc[-200:].mean(),\n"
            "         t1['rate_low_pct'] + t1['planted_spread_pct'],\n"
            "         t1['rate_high_pct'] + t1['planted_spread_pct']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** Implied financing **{R['sso_f']:.3f}% / "
           f"{R['upro_f']:.3f}%** against ^IRX {R['irx_mean']:.3f}% — a "
           f"**+{R['sso_spread']:.2f} pp** mark-up whose own HAC *t* is "
           f"**{R['sso_spread_t']:.2f} / {R['upro_spread_t']:.2f}**, past the |*t*| ≥ 2 bar. "
           f"(The intercept *t* of {R['sso_alpha_t']:.2f} tests only that the drag is "
           f"non-zero — the fee alone guarantees that, so it is not the bar.) Bootstrap CI "
           f"[{R['boot_sso_lo']:+.2f}, {R['boot_sso_hi']:+.2f}] clear of zero; both rate "
           f"regimes clear |*t*| = 2; the whole ER sweep positive; the **all-in** spread, "
           f"free of any fee assumption, at +{R['sso_allin_spread']:.2f} / "
           f"+{R['upro_allin_spread']:.2f} pp (*t* = {R['sso_allin_t']:.2f} / "
           f"{R['upro_allin_t']:.2f}); pass-through slope {R['roll_sso_slope']:.2f}.\n"
           f"- **What the badge does not cover.** (1) **Upper bound, not interest** — swap "
           f"spreads, reset slippage and tracking loss are inside the same residual. "
           f"(2) **The early era is weak** — pre-2018 the spread is "
           f"+{R['era_e_spread']:.2f} at *t* = {R['era_e_t']:.2f}, below the bar; the robust "
           f"finding is the post-2018 +{R['era_l_spread']:.2f}. (3) **Survivorship** — SSO "
           f"and UPRO survived a cohort that lost members and are both ProShares funds on a "
           f"shared swap desk, so their agreement is a consistency check, not independent "
           f"confirmation, and the estimate is a **floor** for the class.\n"
           f"- **Tradability — Investable.** A cost decision with a break-even you can "
           f"compute from your own rate card: **^IRX + {R['be_sso']:.2f}%** at 2x, "
           f"**^IRX + {R['be_upro']:.2f}%** at 3x, never above "
           f"^IRX + {R['be_sso_hiked']:.2f}% in any cut. Above that line the wrapper wins by "
           f"+{R['race_sso_400']:.1f}% to +{R['race_upro_600']:.1f}%/yr — safe by a factor "
           f"of three against retail margin at bills + 4 to 6%, which is why the badge "
           f"survives the wobble in the estimate. (Those races' *t* of 18-39 are mechanical, "
           f"not evidence.) Below the line the margin account wins by "
           f"{R['race_sso_075']:.2f}%/yr — but a prime-broker tier, and the 3x DIY arm "
           f"**Reg T does not permit in a retail margin account**, are not available to the "
           f"holder this rule addresses.\n"
           f"- **Out of scope.** Only the financing is priced here. The daily reset also "
           f"changes the *shape* of the payoff — variance drag, path dependence, the −60% and "
           f"−77% drawdowns — and none of that is netted into these numbers."),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    for name, nb in (("01_for_the_curious.ipynb", build_curious()),
                     ("02_for_the_quants.ipynb", build_quants())):
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path)


if __name__ == "__main__":
    main()
