"""Generate the two narrative notebooks for Study 935 (Value Averaging).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from
the frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells
run the fast synthetic control, which is clearly labelled as synthetic and never
appears under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. SPY (equity sleeve) vs BIL
# (cash leg), total-return, rolling 36-month programmes, 2007-05-30 -> 2026-06-30.
R = dict(
    start="2007-05-30", end="2026-06-30", n_days=4802, fp="9cce1b76d021",
    n_windows=193, first_start="2007-05-31", last_val="2026-06-01",
    horizon=36, buffer=6, growth=0.0, cost_bps=1.0,
    gap=-1.372, gap_med=-1.053, gap_sd=2.113,
    t_hac=-3.87, t_nov=-0.98, n_nov=6, boot_lo=-2.057, boot_hi=-0.742,
    win=29.0, win_lo=23.1, win_hi=35.8,
    exc_va=19.04, exc_dca=20.41,
    irr_prog_va=10.07, irr_prog_dca=10.64,
    irr_eq_va=14.90, irr_eq_dca=13.97, irr_eq_edge=0.92,
    inv_va=0.634, inv_dca=0.699, disp=0.810,
    bind_windows=6, bind_rate=3.1, bind_month=0.20, bind_months_total=14,
    # two DIFFERENT quantities: the largest single-month unfunded call, and the largest
    # sum of unfunded calls within one programme. Never quote the second as the first.
    worst_month_shortfall=3.32, worst_prog_shortfall=6.57,
    va_notional=34.66, dca_notional=36.00,
    worst_gap=-7.38, worst_start="2018-12-31", best_gap=2.87, best_start="2017-04-28",
    era_e_n=104, era_e_gap=-1.518, era_e_t=-2.98, era_e_win=23.1,
    era_l_n=89, era_l_gap=-1.201, era_l_t=-2.50, era_l_win=36.0,
    g0_gap=-1.372, g0_t=-3.87, g0_inv=0.634,
    g4_gap=-0.644, g4_t=-2.02, g4_inv=0.654,
    g8_gap=0.096, g8_t=0.33, g8_inv=0.673,
    g12_gap=0.834, g12_t=3.23, g12_inv=0.693, g12_bind=13.0,
    buf0_gap=-1.675, buf0_t=-5.16, buf0_bind=86.5,
    buf3_gap=-1.421, buf3_bind=6.2,
    buf12_gap=-1.346, buf12_bind=0.0,
    buf24_gap=-1.346, buf24_bind=0.0,
    cost0_gap=-1.372, cost5_gap=-1.370, cost25_gap=-1.363, cost25_t=-3.91,
    h24_gap=-0.481, h24_t=-2.09, h24_win=41.5,
    h60_gap=-5.154, h60_t=-7.10, h60_win=5.9,
    h120_gap=-29.319, h120_t=-28.65, h120_win=0.0,
    em_lambda=0.8984, em_gap=0.702, em_t=3.05, em_win=72.5,
    em_lo=0.269, em_hi=1.078,
    pl_n=12, pl_mean=-0.249, pl_sd=1.441, pl_lo=-2.838, pl_hi=1.805,
    pl_z=0.66, pl_above=2, pl_p=0.231,
    ief_gap=-0.165, ief_t=-1.70, ief_win=48.7,
    qqq_gap=-3.469, qqq_t=-5.04, qqq_win=17.1,
    syn_gap=6.759, syn_t=5.00, syn_win=86.2,
    syn_em_gap=8.185, syn_em_t=4.26, syn_det_gap=-0.114,
)


HEADER = f"""# Study 935 — Value Averaging 📐

**Edleson's rule says: don't invest the same *amount* each month — own the same
*value* each month.** Set a value path in advance, then each month buy or sell
whatever it takes to land on it. The path climbs smoothly, the market does not, so
the rule automatically buys more after a fall and sells after a rally. Its author
reports it beats dollar-cost averaging almost always.

It does — on the metric his book quotes. This study asks what happens when you also
count the **cash the rule needs to exist**: the buffer that funds the extra purchases
in falling markets, and the idle money the rule is *not* investing when the market
runs away from the path.

We test it on **SPY vs BIL** (cash) daily total-return closes, {R['start']} → {R['end']}
({R['n_days']:,} days), over **every** rolling {R['horizon']}-month savings programme
({R['n_windows']} of them), both arms handed identical committed capital, one execution
lag, {R['cost_bps']:.0f} bp one-way.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
live cells run the offline synthetic control only. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The two savers\n\n"
           "Both put aside the same £1 a month for three years. Both start with the same "
           f"reserve of **{R['buffer']} months' contributions** sitting in T-bills. The DCA "
           "saver buys £1 of the market every month and never thinks again. The value-"
           "averaging saver checks a chart each month: if the pot is below the line, buy the "
           "difference; if it is above, **sell** the difference and put the money back in "
           "T-bills.\n\n"
           "At the end we simply ask who has more — counting *everything*, shares and cash."),
        code(
            "R = dict(gap=%r, win=%r, win_lo=%r, win_hi=%r, t_hac=%r,\n"
            "         irr_eq_va=%r, irr_eq_dca=%r, irr_prog_va=%r, irr_prog_dca=%r)\n"
            "print('VA minus DCA, final wealth: %%+.2f cents per pound saved' %% R['gap'])\n"
            "print('value averaging finishes ahead in %%.1f%%%% of the 193 programmes '\n"
            "      '(95%%%% range %%.1f-%%.1f%%%%)' %% (R['win'], R['win_lo'], R['win_hi']))"
            % (R["gap"], R["win"], R["win_lo"], R["win_hi"], R["t_hac"],
               R["irr_eq_va"], R["irr_eq_dca"], R["irr_prog_va"], R["irr_prog_dca"])
        ),
        md("## 2. So why does everyone say it wins?\n\n"
           "Because of *which* return you measure. If you compute the return on the money "
           "**that actually went into shares** — the number the book quotes — value averaging "
           f"really does win: **{R['irr_eq_va']:.2f}%/yr vs {R['irr_eq_dca']:.2f}%/yr**, a "
           f"{R['irr_eq_edge']:+.2f} pp lead. But that number ignores the reserve sitting in "
           "the bank *making the purchases possible*, and it ignores the cash handed back "
           "after every sale. Count the whole account and the ranking flips: "
           f"**{R['irr_prog_va']:.2f}%/yr vs {R['irr_prog_dca']:.2f}%/yr**.\n\n"
           "That is the entire trick. Same two savers, same tape, two different answers — "
           "because one measure quietly leaves out most of one saver's money.\n\n"
           "> 🔬 **For the quants:** the equity-only IRR is near-invariant to the buffer size "
           "across the whole sweep (14.73% at no buffer, 14.91% at 24 months' worth), while "
           "the whole-programme IRR halves over the same range (12.97% → 6.22%). That gap "
           "between the two columns is precisely the tell that the famous number is not "
           "measuring the programme."),
        md("## 3. What is really going on: a hidden dial on how much you own\n\n"
           "Over these nineteen years the market grew faster than the flat value path. So the "
           "value-averaging saver was constantly being told to *sell*, and spent the period "
           f"with only **{R['inv_va']:.1%}** of the account in shares against the DCA saver's "
           f"**{R['inv_dca']:.1%}**. Less in the market, less of the market's return.\n\n"
           "Tilt the value path upward and the gap moves with the equity weight, in lockstep:\n\n"
           f"| assumed path growth | equity weight | VA minus DCA |\n|---|--:|--:|\n"
           f"| 0%/yr (the book's basic path) | {R['g0_inv']:.1%} | {R['g0_gap']:+.2f}c |\n"
           f"| 4%/yr | {R['g4_inv']:.1%} | {R['g4_gap']:+.2f}c |\n"
           f"| 8%/yr | {R['g8_inv']:.1%} | {R['g8_gap']:+.2f}c |\n"
           f"| 12%/yr | {R['g12_inv']:.1%} | {R['g12_gap']:+.2f}c |\n\n"
           "The growth rate of the path is not a detail; it *is* the strategy. And nothing "
           "on the tape tells you what to set it to — you have to guess, in advance, at the "
           "return you are trying to earn."),
        md("## 4. The cash it demands, when you can least spare it\n\n"
           f"With a {R['buffer']}-month reserve the rule ran out of money in "
           f"**{R['bind_windows']} of {R['n_windows']}** programmes — and all six started in "
           "the summer and autumn of **2007**, i.e. straight into the financial crisis. At the "
           f"worst point the rule asked for **{R['worst_month_shortfall']:.1f} extra months of "
           f"contributions in one month** — and **{R['worst_prog_shortfall']:.1f} months' worth "
           "in total** over the programme's binding months — at the exact moment a saver is "
           "least able to find them. Run it with no reserve at all and it is unfundable in "
           f"**{R['buf0_bind']:.0f}%** of programmes.\n\n"
           "The demand is not a rare tail — it is concentrated in precisely the market it is "
           "supposed to be exploiting."),
        md("## 5. Even the fair fight is not evidence of skill\n\n"
           "Give DCA the *same* equity weight (dial its monthly purchase down to "
           f"{R['em_lambda']:.2f}) and value averaging does finally win: "
           f"**{R['em_gap']:+.2f} cents**. But run the identical exercise on a **coin-flip "
           "market** — a random walk with SPY's own drift and volatility, where by construction "
           "there is nothing to predict — and the contrarian schedule earns about the same "
           f"thing anyway (spread {R['pl_lo']:+.2f} to {R['pl_hi']:+.2f} cents across "
           f"{R['pl_n']} such worlds, {R['pl_above']} of which beat the real tape outright). "
           f"The real tape's win sits at z = {R['pl_z']:+.2f} inside that noise. Buying the "
           "dips of a random walk pays a small mechanical bonus; that is all this is."),
        md("## 6. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "If prices genuinely swing back and forth, value averaging *should* clean up. The "
           "cell below plants exactly that and checks the harness finds it — then checks the "
           "harness stays silent on a tape with no swings at all. **Synthetic data, not the "
           "real tape.**"),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from value_avg import data, strategy as st\n"
            "planted, _ = data.synthetic_daily(n_years=12, signal_strength=1.0, seed=935)\n"
            "quiet,   _ = data.synthetic_daily(n_years=12, signal_strength=0.0, seed=935, vol_ann=0.0)\n"
            "pl = st.exposure_matched_race(planted['asset'], planted['cash'], 36,\n"
            "                              tol=0.01, max_iter=6, buffer_mult=6.0, cost_bps=1.0)\n"
            "qt = st.exposure_matched_race(quiet['asset'], quiet['cash'], 36,\n"
            "                              tol=0.005, max_iter=8, buffer_mult=6.0, cost_bps=1.0)\n"
            "print('SYNTHETIC swinging market : VA beats DCA by %+.2f cents (t=%+.2f) -- it works when there is something to work on'\n"
            "      % (pl['gap_mean_cents'], pl['t_hac']))\n"
            "print('SYNTHETIC flat market     : VA beats DCA by %+.2f cents -- nothing to harvest, nothing claimed'\n"
            "      % qt['gap_mean_cents'])"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Value averaging's advantage over dollar-cost averaging does "
           f"not exist on this tape once you count the cash it needs: **{R['gap']:+.2f} cents "
           f"per pound saved** (HAC *t* = {R['t_hac']:+.2f}), ahead in only "
           f"**{R['win']:.0f}%** of programmes, negative in both eras and on both cross-check "
           f"sleeves. The one setting where it wins is the one that quietly raises its equity "
           f"weight, and the exposure-matched residual is indistinguishable from what a "
           f"coin-flip market pays.\n"
           f"- **Tradability — Mirage.** The advertised edge is an accounting choice, not a "
           f"return: the equity-only IRR says **{R['irr_eq_edge']:+.2f} pp/yr** in *every* "
           f"configuration we ran, including the ones where the saver ends up poorer. Costs "
           f"are irrelevant (the rule trades *less* notional than DCA); what is not "
           f"irrelevant is the {R['worst_month_shortfall']:.1f}-months-of-savings cash call it "
           f"can make in a single month during a crash."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 935 — Value Averaging — the teardown\n\n"
           "Rolling 36-month accumulation programmes on **SPY vs BIL**, both arms handed "
           "identical committed capital (a pre-funded buffer plus the same monthly "
           "contributions), idle money earning BIL's actual total return, one execution lag "
           "(sized at the decision month-end close, filled at the next day's close), 1 bp "
           "one-way on traded notional, no shorting and therefore no borrow. The headline "
           "quantity is terminal **whole-account** wealth in cents per dollar contributed.\n\n"
           "Contents: the wealth gap and its HAC *t*, the two IRR measures, the cap-binding "
           "statistics, the era cut, four sweeps (value-path growth, buffer, cost, horizon), "
           "the exposure-matched control, the calibrated random-walk placebo, the sleeve "
           "cross-checks and the live synthetic control. Every real number is frozen from "
           "`docs/results.md` (Fingerprint `%s`)." % R["fp"]),
        code("R = %r" % (R,)),
        md("## The headline\n\n"
           "193 overlapping windows, starts 2007-05-31 → 2023-05-31. The HAC *t* is lag-"
           "truncated at 36 (the overlap span); the block bootstrap uses 36-window blocks; the "
           "non-overlapping check keeps every 36th window and therefore has only 6 "
           "observations, which is reported for honesty rather than for power."),
        code(
            "print(f\"gap (VA - DCA)   : {R['gap']:+.3f}c/dollar   median {R['gap_med']:+.3f}   sd {R['gap_sd']:.3f}\")\n"
            "print(f\"HAC t (lag 36)   : {R['t_hac']:+.2f}   non-overlapping t {R['t_nov']:+.2f} (n={R['n_nov']})\")\n"
            "print(f\"bootstrap 95% CI : [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]  -- entirely below zero\")\n"
            "print(f\"VA win rate      : {R['win']:.1f}%  Wilson [{R['win_lo']:.1f}%, {R['win_hi']:.1f}%]\")\n"
            "print(f\"excess-of-cash   : VA {R['exc_va']:+.2f}c vs DCA {R['exc_dca']:+.2f}c per dollar contributed\")\n"
            "print(f\"mean equity wt   : VA {R['inv_va']:.3f} vs DCA {R['inv_dca']:.3f}   dispersion ratio {R['disp']:.3f}\")\n"
            "print(f\"worst window {R['worst_gap']:+.2f}c ({R['worst_start']})   best {R['best_gap']:+.2f}c ({R['best_start']})\")"
        ),
        md("## The metric that makes VA look good\n\n"
           "Two IRRs on the same programme. `equity_irr` discounts only the flows that touch "
           "the sleeve; `programme_irr` discounts the buffer and the contributions and values "
           "the whole account at the end. They disagree in sign.\n\n"
           "> 💡 **In plain words:** the famous number measures the return on the money that "
           "happened to be invested, not the return on the money you had to commit."),
        code(
            "print(f\"equity-only IRR (Edleson)  : VA {R['irr_eq_va']:.2f}%/yr vs DCA {R['irr_eq_dca']:.2f}%/yr   ({R['irr_eq_edge']:+.2f} pp to VA)\")\n"
            "print(f\"whole-programme IRR        : VA {R['irr_prog_va']:.2f}%/yr vs DCA {R['irr_prog_dca']:.2f}%/yr   ({R['irr_prog_va']-R['irr_prog_dca']:+.2f} pp to VA)\")\n"
            "for b, eq_va, pr_va, pr_dca in [(0, 14.73, 12.97, 13.92), (3, 14.87, 11.35, 12.03),\n"
            "                                (6, 14.90, 10.07, 10.64), (12, 14.91, 8.27, 8.73),\n"
            "                                (24, 14.91, 6.22, 6.56)]:\n"
            "    print(f\"  buffer {b:2d}xC : equity-only VA {eq_va:.2f}% (DCA {R['irr_eq_dca']:.2f}% always)   \"\n"
            "          f\"whole-programme VA {pr_va:5.2f}% vs DCA {pr_dca:5.2f}%\")\n"
            "print('the left column barely moves while the programme underneath it changes completely')"
        ),
        md("## The mechanism — it is an exposure dial\n\n"
           "The value path's growth rate is an **ASSUMPTION**, nowhere on the tape. Sweeping "
           "it moves VA's average equity weight, and the wealth gap follows it monotonically; "
           "the sign flips at roughly the point where the weights match. That is not a "
           "strategy parameter, it is a beta parameter."),
        code(
            "for g, gap, t, inv in [(0, R['g0_gap'], R['g0_t'], R['g0_inv']),\n"
            "                       (4, R['g4_gap'], R['g4_t'], R['g4_inv']),\n"
            "                       (8, R['g8_gap'], R['g8_t'], R['g8_inv']),\n"
            "                       (12, R['g12_gap'], R['g12_t'], R['g12_inv'])]:\n"
            "    print(f\"path growth {g:2d}%/yr : VA equity {inv:.3f} (DCA {R['inv_dca']:.3f})  gap {gap:+.3f}c  t {t:+.2f}\")"
        ),
        md("## The funding constraint — how often the cap binds\n\n"
           "The buffer is finite and the purchase is capped at what it can fund; the shortfall "
           "is recorded rather than silently borrowed. The binding is rare on average and "
           "brutally concentrated: all six binding programmes start between 2007-05 and "
           "2007-10."),
        code(
            "print(f\"buffer {R['buffer']}xC : binds in {R['bind_windows']}/{R['n_windows']} windows ({R['bind_rate']:.1f}%), {R['bind_months_total']} binding months = {R['bind_month']:.2f}% of all rebalance months\")\n"
            "print(f\"worst SINGLE-MONTH unfunded call : {R['worst_month_shortfall']:.2f} x the monthly contribution\")\n"
            "print(f\"worst PROGRAMME-TOTAL shortfall  : {R['worst_prog_shortfall']:.2f} x (summed over that programme's binding months)\")\n"
            "for b, gap, bind in [(0, R['buf0_gap'], R['buf0_bind']), (3, R['buf3_gap'], R['buf3_bind']),\n"
            "                     (6, R['gap'], R['bind_rate']), (12, R['buf12_gap'], R['buf12_bind']),\n"
            "                     (24, R['buf24_gap'], R['buf24_bind'])]:\n"
            "    print(f\"  buffer {b:2d}xC : gap {gap:+.3f}c   cap binds in {bind:.1f}% of windows\")\n"
            "print('no buffer setting rescues the sign; a bigger buffer only dilutes both arms equally')"
        ),
        md("## Robustness — eras, costs, horizons, sleeves\n\n"
           "Costs are a non-issue: VA trades *less* notional than DCA "
           f"({R['va_notional']:.1f} vs {R['dca_notional']:.1f} per 36 dollars contributed), "
           "because its sells net against its buys, so the gap barely moves out to 25 bp. The "
           "horizon sweep is the clearest reading of the mechanism — the longer the programme, "
           "the more the compounding equity-weight shortfall dominates."),
        code(
            "print(f\"era, start <2016 (n={R['era_e_n']}): gap {R['era_e_gap']:+.3f}c  t {R['era_e_t']:+.2f}  win {R['era_e_win']:.1f}%\")\n"
            "print(f\"era, start >=2016 (n={R['era_l_n']}): gap {R['era_l_gap']:+.3f}c  t {R['era_l_t']:+.2f}  win {R['era_l_win']:.1f}%\")\n"
            "print(f\"cost 0/5/25 bp : {R['cost0_gap']:+.3f} / {R['cost5_gap']:+.3f} / {R['cost25_gap']:+.3f}c (t {R['cost25_t']:+.2f} at 25bp)\")\n"
            "for h, gap, t, w in [(24, R['h24_gap'], R['h24_t'], R['h24_win']), (36, R['gap'], R['t_hac'], R['win']),\n"
            "                     (60, R['h60_gap'], R['h60_t'], R['h60_win']), (120, R['h120_gap'], R['h120_t'], R['h120_win'])]:\n"
            "    print(f\"  horizon {h:3d}m : gap {gap:+8.3f}c  t {t:+7.2f}  VA wins {w:5.1f}%\"\n"
            "          + ('   <- t is DIRECTION only: HAC lag = horizon, and only 3 (60m) / 1 (120m) independent programmes exist' if h >= 60 else ''))\n"
            "print(f\"IEF sleeve : gap {R['ief_gap']:+.3f}c (t {R['ief_t']:+.2f})   QQQ sleeve : gap {R['qqq_gap']:+.3f}c (t {R['qqq_t']:+.2f})\")\n"
            "print('the gap scales with the sleeve return -- lowest on bonds, worst on the highest-drift sleeve')"
        ),
        md("## The decisive cut — exposure-matched, then placebo'd\n\n"
           "Dial DCA's monthly purchase down by lambda until the two arms carry the same mean "
           "equity weight; whatever survives is the contrarian *timing*. It is positive. "
           "**lambda is fitted in-sample on this very tape**, so that figure is an in-sample "
           "residual, not an out-of-sample result. Then run the identical exercise — same "
           "in-sample bisection — on twelve random walks calibrated to SPY's own drift, "
           "volatility and cash rate, i.e. worlds with **zero** predictability, and the same "
           "residual appears, because buying the dips of a random walk earns a mechanical "
           "rebalancing bonus. The real tape's residual sits inside that placebo spread. "
           "(The placebo tapes are homoskedastic, so if anything they *understate* the bonus a "
           "vol-clustered tape would hand a contrarian schedule — which cuts against, not for, "
           "the reading below.)"),
        code(
            "print(f\"exposure-matched (lambda={R['em_lambda']:.4f}, fitted IN-SAMPLE): gap {R['em_gap']:+.3f}c  t {R['em_t']:+.2f}  \"\n"
            "      f\"win {R['em_win']:.1f}%  CI [{R['em_lo']:+.3f}, {R['em_hi']:+.3f}]\")\n"
            "print(f\"calibrated random-walk placebo ({R['pl_n']} paths): mean {R['pl_mean']:+.3f}c  sd {R['pl_sd']:.3f}  \"\n"
            "      f\"range [{R['pl_lo']:+.3f}, {R['pl_hi']:+.3f}]\")\n"
            "print(f\"-> the real tape's residual sits at z = {R['pl_z']:+.2f} of the no-predictability spread\")\n"
            "print(f\"-> distribution-free: {R['pl_above']}/{R['pl_n']} zero-predictability paths beat it (one-sided p = {R['pl_p']:.3f})\")"
        ),
        md("## Live synthetic control — the machinery is unbiased\n\n"
           "Planted transitory mean reversion (a sub-one variance ratio by construction): the "
           "exposure-matched race MUST find it. Zero-volatility tape: it must find nothing. "
           "**Synthetic data, not the real tape** — this proves the harness works, it never "
           "supports the stamp."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from value_avg import data, strategy as st\n"
            "planted, truth = data.synthetic_daily(n_years=12, signal_strength=1.0, seed=935)\n"
            "quiet,   _     = data.synthetic_daily(n_years=12, signal_strength=0.0, seed=935, vol_ann=0.0)\n"
            "pl = st.exposure_matched_race(planted['asset'], planted['cash'], 36, tol=0.01,\n"
            "                              max_iter=6, buffer_mult=6.0, cost_bps=1.0)\n"
            "qt = st.exposure_matched_race(quiet['asset'], quiet['cash'], 36, tol=0.005,\n"
            "                              max_iter=8, buffer_mult=6.0, cost_bps=1.0)\n"
            "print('SYNTHETIC planted wobble (swing sd %.2f, half-life %.0fd):' % (truth['swing_eff'], truth['half_life_days']))\n"
            "print('   exposure-matched gap %+.3fc  t %+.2f  VA wins %.1f%%' % (pl['gap_mean_cents'], pl['t_hac'], pl['va_win_rate']*100))\n"
            "print('SYNTHETIC zero-volatility null:')\n"
            "print('   exposure-matched gap %+.3fc  (must be ~0)' % qt['gap_mean_cents'])"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** On identical committed capital the value-averaging programme "
           f"ends **{R['gap']:+.3f} cents per dollar contributed** behind plain DCA (HAC *t* = "
           f"{R['t_hac']:+.2f}, bootstrap CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}] entirely "
           f"below zero, ahead in {R['win']:.1f}% of {R['n_windows']} programmes, negative in "
           f"both eras, on IEF ({R['ief_gap']:+.3f}) and on QQQ ({R['qqq_gap']:+.3f}), and at "
           f"every horizon from 24 to 120 months). The sign is set by a **non-tape assumption** "
           f"— the value path's growth rate, which is simply an equity-weight dial "
           f"({R['g0_inv']:.3f} at 0%/yr to {R['g12_inv']:.3f} at 12%/yr against DCA's "
           f"{R['inv_dca']:.3f}). Exposure-matched on an **in-sample** lambda, the residual is "
           f"**{R['em_gap']:+.3f}c** — inside the spread a calibrated random walk produces with "
           f"no predictability at all (z = {R['pl_z']:+.2f}; {R['pl_above']} of {R['pl_n']} such "
           f"worlds beat it outright, one-sided *p* = {R['pl_p']:.2f}), i.e. the rebalancing "
           f"bonus, not timing. The synthetic control recovers a planted wobble "
           f"({R['syn_em_gap']:+.2f}c, *t* = {R['syn_em_t']:+.2f}) and is silent on a "
           f"zero-vol tape ({R['syn_det_gap']:+.3f}c), so the harness is not the problem.\n"
           f"- **Tradability — Mirage.** The advertised edge survives only in the equity-only "
           f"IRR ({R['irr_eq_edge']:+.2f} pp/yr), a figure near-invariant to the buffer size "
           f"(14.73% to 14.91% across the 0-24x sweep) while the whole-programme IRR halves "
           f"over the same range — so the famous number cannot be measuring the programme; at "
           f"the 6x default it is {R['irr_prog_va']:.2f}% vs {R['irr_prog_dca']:.2f}%. Friction "
           f"is not the obstacle (VA trades less notional than DCA and the gap is unchanged at "
           f"25 bp); the obstacles are the equity-weight give-up and a cash call that reached "
           f"{R['worst_month_shortfall']:.1f} monthly contributions in a single month "
           f"({R['worst_prog_shortfall']:.1f} across that programme), in 2008, when it was "
           f"least payable."),
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
