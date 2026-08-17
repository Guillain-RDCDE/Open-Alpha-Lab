"""Generate the two narrative notebooks for Study 944 (How Much Leverage).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every **real-tape** number is quoted from
the frozen ``R`` dict below, which mirrors ``docs/results.md`` exactly; the only live
cells run the fast offline **synthetic control**, and they are always introduced as such
— no synthetic output ever appears under a real-tape banner.
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
# SPY (total return) financed at ^IRX + 50 bps, daily reset, 1 bp one-way,
# 2003-06-04 -> 2026-06-30, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2003-06-04", end="2026-06-30", n_days=5799, fp="d6dfd514b42d",
    spread_bps=50.0, cost_bps=1.0,
    irx_ann=1.47, bil_ann=1.36, xcheck_gap_bps=11,
    # the curve
    curve_lev=[1.00, 1.50, 2.00, 2.50, 2.85, 3.00],
    curve_tw=[11.68, 22.81, 36.35, 47.17, 49.99, 49.66],
    curve_cagr=[11.27, 14.56, 16.90, 18.23, 18.53, 18.49],
    curve_g=[10.68, 13.59, 15.61, 16.75, 17.00, 16.97],
    curve_sharpe=[0.575, 0.566, 0.560, 0.557, 0.555, 0.555],
    curve_vol=[18.6, 27.8, 37.1, 46.4, 52.9, 55.7],
    curve_dd=[-55.2, -72.6, -84.2, -91.3, -94.5, -95.5],
    curve_turn=[0.0, 1.4, 3.8, 7.2, 10.1, 11.5],
    opt=2.85, kelly=3.10,
    sharpe_gross_l1=0.5752, sharpe_gross_l3=0.5752,
    # (1) bootstrap of the argmax
    boot_ci_lo=1.00, boot_ci_hi=3.00, boot_sd=0.60,
    boot_at_floor=2.9, boot_at_cap=44.4, boot_n=1000, boot_block=63,
    # (2) rolling five-year hindsight optimum
    roll_n=217, roll_mean=2.36, roll_sd=0.86, roll_min=1.00, roll_max=3.00,
    roll_at_floor=24.0, roll_at_cap=54.4, roll_kelly_min=-1.53, roll_kelly_max=10.82,
    roll_years=[2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    roll_vals=[1.00, 1.00, 1.00, 1.00, 1.00, 3.00, 3.00, 3.00, 3.00, 3.00,
               3.00, 3.00, 3.00, 3.00, 2.05, 2.85, 2.80, 3.00, 3.00],
    # (3) era cut and the hand-off
    era_e_n=2911, era_e_opt=2.20, era_e_kelly=2.37, era_e_cagr_opt=11.93,
    era_e_cagr_l1=8.82, era_e_dd_l1=-55.2, era_e_dd_l3=-95.5,
    era_l_n=2888, era_l_opt=3.00, era_l_kelly=3.99, era_l_cagr_opt=27.07,
    era_l_cagr_l1=13.80, era_l_dd_l1=-33.7, era_l_dd_l3=-76.2,
    handoff_late_in_early=10.56, handoff_early_unlev=8.82, handoff_late_in_early_dd=-95.5,
    handoff_early_in_late=23.52, handoff_late_opt_cagr=27.07,
    # (4) start-date sensitivity — the sharpest number in the study
    ss_starts=["2003-06-04", "2004-01-06", "2005-01-03", "2007-01-03", "2010-01-04"],
    ss_n=[5799, 5651, 5401, 4901, 4145],
    ss_opt=[2.85, 2.75, 2.65, 2.60, 3.00],
    ss_kelly=[3.10, 2.94, 2.88, 2.79, 4.55],
    ss_cagr_opt=[18.53, 17.03, 16.86, 16.80, 31.00],
    ss_handoff=[10.56, 7.07, 5.67, 2.66, 40.49],
    ss_unlev=[8.82, 7.81, 7.63, 6.98, 15.38],
    ss_edge=[1.73, -0.74, -1.96, -4.32, 25.11],
    # the tradable arm
    kel_start="2006-06-08", kel_end="2026-06-30", kel_n=5043,
    kel_mean_lev=2.53, kel_at_cap=64.0, kel_at_floor=16.8,
    kel_tw=33.95, kel_cagr=19.26, kel_sharpe=0.597, kel_vol=41.2, kel_dd=-71.6,
    l1_tw=8.61, l1_cagr=11.36, l1_sharpe=0.567, l1_vol=19.4, l1_dd=-55.2,
    l2_tw=22.35, l2_cagr=16.79, l2_sharpe=0.553, l2_dd=-84.2,
    adv_vs1=6.86, t_vs1=1.20, ci_vs1_lo=-5.56, ci_vs1_hi=17.52, t_arith_vs1=2.54,
    adv_vs2=2.09, t_vs2=0.56,
    # the one place a time-varying multiple CAN move Sharpe
    sh_diff=0.030, sh_ci_lo=-0.180, sh_ci_hi=0.254, sh_frac_pos=59,
    caps=[1.5, 2.0, 2.5, 3.0], cap_adv=[2.78, 4.63, 5.91, 6.86],
    cap_t=[1.74, 1.46, 1.30, 1.20], cap_dd=[-60.3, -65.3, -68.9, -71.6],
    wins=[252, 504, 756, 1260], win_adv=[3.46, 4.95, 6.86, 5.12],
    win_t=[0.71, 0.93, 1.20, 0.78],
    # PROXY sweeps
    spreads=[0, 25, 50, 100, 200], spread_opt=[3.00, 2.95, 2.85, 2.75, 2.45],
    spread_cagr=[19.71, 19.11, 18.53, 17.45, 15.56],
    spread_sharpe=[0.573, 0.564, 0.555, 0.539, 0.509],
    costs=[0.0, 1.0, 5.0], cost_opt=[2.90, 2.85, 2.80],
    cost_cagr=[18.66, 18.53, 18.06],
    # synthetic control (frozen mirror; the notebooks also re-run it live)
    syn_planted_mean=1.91, syn_null_mean=0.44,
    syn_planted_kelly=2.04, syn_null_kelly=0.01,
    syn_cond_argmax=3.05, syn_cond_kelly=3.04,
)


HEADER = f"""# Study 944 — How Much Leverage ⚖️

**Constant leverage on the S&P 500: where is the growth-optimal multiple, and could you
ever have known?**

Hold the index at a fixed multiple `L`, reset daily, financing the borrowed part at
T-bills plus a spread. Growth rises with `L`, then the variance drag `L²σ²/2` swamps it —
so there is a peak, and theory names it: the Kelly / Merton multiple `L* = μ/σ²`. The
practical question is not whether that peak exists. It is whether its *location* is a
number you can estimate today and use tomorrow.

Tape: **SPY** daily **total-return** closes financed at **^IRX** + **{R['spread_bps']:.0f} bps/yr
(a PROXY, swept 0-200)**, {R['cost_bps']:.0f} bp one-way on the daily reset's turnover,
{R['start']} → {R['end']} ({R['n_days']:,} days). Every Sharpe is excess-of-cash on both sides.

*Real-tape numbers below are the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`, as-of 2026-06-30). The live cells run the **offline synthetic control** only,
and say so.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Why there is a 'best' amount of leverage at all\n\n"
           "Double your exposure and you double your average daily return — but you also "
           "double your daily swings, and swings *cost* you when you compound. Lose 50% "
           "and you need +100% to get level. So the growth of a levered index goes up, "
           "flattens, and eventually turns back down. Somewhere there is a peak.\n\n"
           "Here is the real curve over 2003-2026, after paying to borrow and after the "
           "daily reset's trading costs."),
        code(
            "R = dict(lev=%r, tw=%r, cagr=%r, dd=%r, sharpe=%r, opt=%r)\n"
            "print(f\"{'leverage':>9}  {'$1 became':>10}  {'growth/yr':>10}  \"\n"
            "      f\"{'worst loss':>11}  {'Sharpe':>7}\")\n"
            "for i, L in enumerate(R['lev']):\n"
            "    mark = '  <-- the peak' if L == R['opt'] else ''\n"
            "    print(f\"{L:9.2f}x  {R['tw'][i]:9.2f}x  {R['cagr'][i]:+9.2f}%%  \"\n"
            "          f\"{R['dd'][i]:10.1f}%%  {R['sharpe'][i]:+7.3f}{mark}\")"
            % (R["curve_lev"], R["curve_tw"], R["curve_cagr"], R["curve_dd"],
               R["curve_sharpe"], R["opt"])
        ),
        md(f"## 2. The peak is real — and so is its price\n\n"
           f"The best multiple over these 23 years was **{R['opt']:.2f}×**: $1 became "
           f"**${R['curve_tw'][4]:.2f}** instead of ${R['curve_tw'][0]:.2f} unlevered. "
           f"Textbook Kelly, computed on the same tape, says **{R['kelly']:.2f}×** — close, "
           f"and slightly higher, because the formula ignores the borrowing spread and the "
           f"fat left tail.\n\n"
           f"Now read the last two columns of that table again. At the peak your worst "
           f"loss was **{R['curve_dd'][4]:.1f}%** — the account is down to five cents on "
           f"the dollar — and the **Sharpe ratio actually got slightly worse** "
           f"({R['curve_sharpe'][0]:.3f} → {R['curve_sharpe'][4]:.3f}). That is not a "
           f"coincidence or a quirk of this sample: constant leverage multiplies your "
           f"excess returns *and* your excess risk by the same number, so it cannot improve "
           f"risk-adjusted return. All it can do is move you along the same line, and pay "
           f"the financing on the way.\n\n"
           f"> 🔬 **For the quants:** gross of the financing spread the excess Sharpe is "
           f"{R['sharpe_gross_l1']:.4f} at 1× and {R['sharpe_gross_l3']:.4f} at 3× — "
           f"identical to four decimals, because `e_L = L·(r − r_f)` exactly. The Sharpe "
           f"axis is degenerate by construction; only geometric growth can distinguish the "
           f"multiples."),
        md(f"## 3. So: is {R['opt']:.2f}× the answer?\n\n"
           f"No — and this is the whole study. Resample the 23 years in quarterly blocks "
           f"and re-solve for the best multiple each time, {R['boot_n']:,} times over. The "
           f"answers fill the **entire grid**: the 95% interval runs "
           f"**[{R['boot_ci_lo']:.2f}, {R['boot_ci_hi']:.2f}]**. "
           f"{R['boot_at_cap']:.0f}% of resamples say 'as much as you'll let me', "
           f"{R['boot_at_floor']:.0f}% say 'none at all'.\n\n"
           f"Twenty-three years of daily data — about 5,800 observations — and we cannot say "
           f"whether the right answer is 1× or 3×. The reason is simple arithmetic: the "
           f"optimum is `average return ÷ variance`, and while variance is easy to measure, "
           f"the *average return* is the single hardest number to pin down in finance."),
        md(f"## 4. And it moves\n\n"
           f"Take a five-year window, look back at it with **perfect hindsight**, and ask "
           f"what leverage *would* have been best. Slide that window through history "
           f"({R['roll_n']} of them):\n\n"
           f"- **{R['roll_at_floor']:.0f}%** of windows: the answer is 1.00× — don't lever.\n"
           f"- **{R['roll_at_cap']:.0f}%** of windows: the answer is 3.00× — lever as hard "
           f"as we allow.\n\n"
           f"Year-end readings run 1.00 through 2008-2012 (the financial crisis is inside "
           f"the window), then flip to 3.00 for most of the decade that follows. The "
           f"underlying Kelly number swings from **{R['roll_kelly_min']:+.1f}** to "
           f"**{R['roll_kelly_max']:+.1f}**. This is the *easy* version of the question — "
           f"cheating with hindsight — and it still has no stable answer."),
        code(
            "years = %r\nvals = %r\n"
            "print('the best leverage, in hindsight, over the previous five years:')\n"
            "for y, v in zip(years, vals):\n"
            "    bar = '#' * int(round(v * 12))\n"
            "    print(f'  {y}  {v:4.2f}x  {bar}')"
            % (R["roll_years"], R["roll_vals"])
        ),
        md(f"## 5. The decade hand-off, and the number that really settles it\n\n"
           f"Split the sample in 2015. The first era's best multiple was "
           f"**{R['era_e_opt']:.2f}×**; the second era's was **{R['era_l_opt']:.2f}×** "
           f"(capped — it wanted more). Take one era's answer and use it in the other: the "
           f"2015-2026 optimum ({R['era_l_opt']:.2f}×) applied to 2003-2014 earned "
           f"{R['handoff_late_in_early']:+.2f}%/yr against "
           f"{R['handoff_early_unlev']:+.2f}%/yr for not levering at all. It stayed ahead "
           f"— by 1.7 points — while losing {R['handoff_late_in_early_dd']:.1f}% peak to "
           f"trough, which is to say the account was gone long before the compounding "
           f"argument could pay out.\n\n"
           f"But here is the number that actually settles it, and it is an uncomfortable "
           f"one. **Change nothing except where the sample starts.** Same code, same end "
           f"date, same instrument — only the left edge of the window moves:"),
        code(
            f"starts = {R['ss_starts']!r}\n"
            f"opt = {R['ss_opt']!r}\n"
            f"hand = {R['ss_handoff']!r}\n"
            f"unlev = {R['ss_unlev']!r}\n"
            f"edge = {R['ss_edge']!r}\n"
            "print('sample starts   best leverage   its answer used in the other era   "
            "vs not levering')\n"
            "for i, s0 in enumerate(starts):\n"
            "    verdict = 'BEAT it' if edge[i] > 0 else 'LOST to it'\n"
            "    print(f'{s0:>13}   {opt[i]:12.2f}x   {hand[i]:+29.2f}%   "
            "{unlev[i]:+14.2f}%   -> {verdict} by {abs(edge[i]):.2f}%/yr')"
        ),
        md(f"Read the last column. Starting the sample in June 2003, one decade's optimal "
           f"leverage still **beat** not levering in the other decade. Starting it in "
           f"January 2004 — **seven months later**, on a boundary nobody chose for any "
           f"economic reason — it **lost**. Start in 2007 and it loses by "
           f"{abs(R['ss_edge'][3]):.2f} points a year. Start in 2010, after the crash has "
           f"been deleted from the sample, and the tape cheerfully reports that the best "
           f"leverage is {R['ss_opt'][4]:.2f}× with a Kelly of {R['ss_kelly'][4]:.2f} — "
           f"lever four and a half times, says the data, because nothing bad has happened "
           f"yet.\n\n"
           f"'Optimal leverage' is not a property of the market. It is a property of the "
           f"slice of history you happened to load."),
        md(f"## 6. The tradable version does not rescue it\n\n"
           f"Estimate `μ/σ²` from the trailing three years, act on it the next day, cap at "
           f"3×. It compounds at **{R['kel_cagr']:.2f}%/yr** against "
           f"**{R['l1_cagr']:.2f}%** for plain buy-and-hold — an advantage of "
           f"**{R['adv_vs1']:+.2f}%/yr**. But the statistical test on that growth gap gives "
           f"***t* = {R['t_vs1']:+.2f}** with a confidence interval of "
           f"[{R['ci_vs1_lo']:+.1f}%, {R['ci_vs1_hi']:+.1f}%] — it comfortably includes "
           f"zero, and the desk's bar is |*t*| ≥ 2.\n\n"
           f"Worse, the rule spends **{R['kel_at_cap']:.0f}% of its days pinned at the 3× "
           f"cap**. Loosen the cap and the return rises while the *t*-stat *falls* "
           f"({R['cap_adv'][0]:+.2f}%/yr at cap 1.5 → {R['cap_adv'][3]:+.2f}% at cap 3.0; "
           f"*t* {R['cap_t'][0]:+.2f} → {R['cap_t'][3]:+.2f}). That is the signature of a "
           f"volume knob, not a signal."),
        md("## 7. Live check — the machinery is honest (offline synthetic)\n\n"
           "The cell below is **synthetic, not the real tape**. It builds a make-believe "
           "market whose best leverage is *known* to be 2.0, and a second one where the "
           "asset earns exactly cash so any leverage is pure waste. Our sweep must find "
           "2.0 in the first and the floor in the second — otherwise the real-tape result "
           "would just be a broken tool.\n\n"
           "Watch the individual seeds, though. Even in this stationary, made-up world, "
           "*forty years* of daily data locates the optimum only to about ±1. That is the "
           "study's finding in a laboratory."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from optimal_leverage import data, strategy as st\n"
            "GRID = np.round(np.arange(0.0, 3.0001, 0.25), 4)\n"
            "for tag, ss in [('planted best = 2.0', 1.0), ('null (no reward)', 0.0)]:\n"
            "    opts = [st.synthetic_detect(data.synthetic_daily(signal_strength=ss, seed=944+s)[0],\n"
            "                                grid=GRID)['opt_lev'] for s in range(8)]\n"
            "    print(f'{tag:20s} mean {np.mean(opts):.2f}   per-seed {sorted(opts)}')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** Levering did raise realised growth, in both eras and at "
           f"every cost assumption, and the curve peaks sensibly ({R['opt']:.2f}× against a "
           f"Kelly of {R['kelly']:.2f}×). But the claim under test is that the peak is a "
           f"*locatable* number, and it is not: the bootstrap interval for it is the whole "
           f"grid, the hindsight answer oscillates between 1× and 3×, and the tradable "
           f"version clears no significance bar at any setting.\n"
           f"- **Tradability — Mirage.** Nothing risk-adjusted is on offer — leverage cannot "
           f"move the Sharpe ratio, only lower it through financing. The growth is paid for "
           f"with a {R['curve_dd'][4]:.0f}% drawdown, and the number you would need to know "
           f"is the one number the data refuses to tell you — it changes sign on "
           f"seven months of start date.\n"
           f"- **What is honestly true.** The *shape* of the curve is robust. Its *location* "
           f"is not, and the location is the entire practical question."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 944 — How Much Leverage — the teardown\n\n"
           "`r_L = L·r_SPY − (L−1)·(^IRX + spread) − cost·turnover`, reset daily. "
           "The g(L) curve against Kelly `μ/σ²`, the Sharpe-invariance identity, a "
           "block bootstrap of the **argmax**, the rolling five-year hindsight optimum, "
           "the era hand-off, the **start-date sweep**, the ex-ante Kelly race tested on "
           "**log**-return differences, "
           "the two PROXY sweeps, and the live synthetic control. Every real number is "
           "frozen from `docs/results.md` (Fingerprint `%s`, as-of 2026-06-30); the only "
           "live cells are explicitly synthetic." % R["fp"]),
        code("R = %r" % (R,)),
        md("## 0. Provenance and the two assumptions\n\n"
           "SPY total-return closes (`auto_adjust=True`) financed at ^IRX (13-week bill "
           "**discount** rate, act/360 on the previous close, accrued over the calendar "
           "days the bar spans). ^IRX rather than BIL because it covers the whole window; "
           "the two are cross-checked below. Two non-tape inputs, both **PROXIES**, both "
           "swept: the financing spread over bills and the one-way reset cost.\n\n"
           "> 💡 **In plain words:** we are not measuring what it costs to borrow — we are "
           "*assuming* it, then checking how much the answer depends on the assumption."),
        code(
            "print(f\"window {R['start']} -> {R['end']}   n={R['n_days']:,}   fp={R['fp']}\")\n"
            "print(f\"financing cross-check 2007-2026: ^IRX-implied {R['irx_ann']:.2f}%/yr vs \"\n"
            "      f\"BIL total return {R['bil_ann']:.2f}%/yr  (gap {R['xcheck_gap_bps']:+d} bps/yr)\")\n"
            "print(f\"PROXY spread {R['spread_bps']:.0f} bps/yr (swept 0-200) | \"\n"
            "      f\"PROXY reset cost {R['cost_bps']:.0f} bp one-way (swept 0-5)\")\n"
            "print('one execution lag, used exactly once: the ex-ante Kelly estimate (through t, applied t+1)')"
        ),
        md("## 1. The g(L) curve, and the Sharpe identity that kills one axis\n\n"
           "`g(L) ≈ L·μ − L²σ²/2 − (L−1)·s`, concave, peaking at `(μ−s)/σ²`. The excess "
           "return is `L·(r − r_f) − (L−1)·s − c`, so **gross of s and c the Sharpe is "
           "exactly invariant in L**. Reported below to four decimals as a construction "
           "check, not a finding."),
        code(
            "print(f\"{'L':>5}  {'terminal':>9}  {'CAGR':>8}  {'g(L)':>8}  {'exSharpe':>9}  \"\n"
            "      f\"{'vol':>7}  {'maxDD':>8}  {'turn/yr':>8}\")\n"
            "for i, L in enumerate(R['curve_lev']):\n"
            "    mark = '  <-- realised optimum' if L == R['opt'] else ''\n"
            "    print(f\"{L:5.2f}  {R['curve_tw'][i]:8.2f}x  {R['curve_cagr'][i]:+7.2f}%  \"\n"
            "          f\"{R['curve_g'][i]:+7.2f}%  {R['curve_sharpe'][i]:+9.3f}  \"\n"
            "          f\"{R['curve_vol'][i]:6.1f}%  {R['curve_dd'][i]:+7.1f}%  \"\n"
            "          f\"{R['curve_turn'][i]:7.1f}x{mark}\")\n"
            "print(f\"\\nrealised argmax {R['opt']:.2f}   theoretical Kelly mu/sigma^2 {R['kelly']:.2f}\")\n"
            "print(f\"Sharpe invariance (0 bps spread, 0 cost): L=1 {R['sharpe_gross_l1']:.4f}  \"\n"
            "      f\"L=3 {R['sharpe_gross_l3']:.4f}  -> identical by construction\")"
        ),
        md(f"The realised peak sits **below** Kelly ({R['opt']:.2f} vs {R['kelly']:.2f}), as "
           f"it must: the closed form assumes Gaussian returns and free financing, and both "
           f"the fat left tail and the {R['spread_bps']:.0f} bps spread push the optimum "
           f"down. Directionally correct machinery — which is the only thing this comparison "
           f"is allowed to establish."),
        md("## 2. Is the argmax identified? Block bootstrap\n\n"
           "Circular block bootstrap, %d draws, %d-day blocks of the joint `(r_asset, "
           "r_cash)` rows so vol clustering survives; the **whole grid is re-solved** on "
           "each resample and the argmax recorded.\n\n"
           "> 💡 **In plain words:** we shuffle history in quarterly chunks and ask, a "
           "thousand times, 'what leverage would have been best?' If the answer were "
           "knowable, the thousand answers would cluster."
           % (R["boot_n"], R["boot_block"])),
        code(
            "print(f\"argmax {R['opt']:.2f}   95% CI [{R['boot_ci_lo']:.2f}, {R['boot_ci_hi']:.2f}]  \"\n"
            "      f\"sd {R['boot_sd']:.2f}\")\n"
            "print(f\"draws at the 1.00 floor: {R['boot_at_floor']:.1f}%   \"\n"
            "      f\"draws at the 3.00 cap: {R['boot_at_cap']:.1f}%\")\n"
            "print('-> the CI is the entire grid: the location of the peak is not identified')"
        ),
        md(f"This is not estimator weakness, it is the arithmetic of `L* = μ/σ²`. With "
           f"σ ≈ {R['curve_vol'][0]:.0f}%/yr and 23.1 years, `se(μ) ≈ "
           f"{R['curve_vol'][0]/100/23.1**0.5*100:.1f}%/yr`, so `se(L*) ≈ "
           f"{(R['curve_vol'][0]/100/23.1**0.5)/(R['curve_vol'][0]/100)**2:.1f}` — Merton "
           f"(1980) in one line. Variance converges with sampling frequency; the mean only "
           f"with calendar span."),
        md("## 3. Is the argmax stable? Rolling five-year hindsight optimum\n\n"
           "%d windows of 1,260 days, monthly stride, each solved with **perfect hindsight "
           "inside the window** — the easiest possible version of the problem."
           % R["roll_n"]),
        code(
            "print(f\"{R['roll_n']} windows: mean {R['roll_mean']:.2f}  sd {R['roll_sd']:.2f}  \"\n"
            "      f\"range [{R['roll_min']:.2f}, {R['roll_max']:.2f}]\")\n"
            "print(f\"at the 1.00 floor {R['roll_at_floor']:.1f}% of windows  |  \"\n"
            "      f\"at the 3.00 cap {R['roll_at_cap']:.1f}%\")\n"
            "print(f\"rolling Kelly estimate ranges [{R['roll_kelly_min']:+.2f}, {R['roll_kelly_max']:+.2f}]\")\n"
            "print('\\nyear-end reading:')\n"
            "for y, v in zip(R['roll_years'], R['roll_vals']):\n"
            "    print(f'  {y}  {v:4.2f}x  ' + '#' * int(round(v * 12)))"
        ),
        md("## 4. The era cut, and the hand-off test\n\n"
           "Split 2015-01-01. A stable optimum should survive the split; the hand-off asks "
           "the practical question directly — take one era's answer, use it in the other."),
        code(
            "print(f\"early n={R['era_e_n']}: optimum {R['era_e_opt']:.2f}  Kelly {R['era_e_kelly']:.2f}  \"\n"
            "      f\"CAGR@opt {R['era_e_cagr_opt']:+.2f}%  CAGR@1 {R['era_e_cagr_l1']:+.2f}%  \"\n"
            "      f\"DD@1 {R['era_e_dd_l1']:+.1f}%  DD@3 {R['era_e_dd_l3']:+.1f}%\")\n"
            "print(f\"late  n={R['era_l_n']}: optimum {R['era_l_opt']:.2f}  Kelly {R['era_l_kelly']:.2f}  \"\n"
            "      f\"CAGR@opt {R['era_l_cagr_opt']:+.2f}%  CAGR@1 {R['era_l_cagr_l1']:+.2f}%  \"\n"
            "      f\"DD@1 {R['era_l_dd_l1']:+.1f}%  DD@3 {R['era_l_dd_l3']:+.1f}%\")\n"
            "print()\n"
            "print(f\"hand-off: late optimum {R['era_l_opt']:.2f}x applied 2004-2014 -> \"\n"
            "      f\"{R['handoff_late_in_early']:+.2f}%/yr vs {R['handoff_early_unlev']:+.2f}%/yr unlevered \"\n"
            "      f\"(DD {R['handoff_late_in_early_dd']:+.1f}%)\")\n"
            "print(f\"          early optimum {R['era_e_opt']:.2f}x applied 2015-2026 -> \"\n"
            "      f\"{R['handoff_early_in_late']:+.2f}%/yr vs {R['handoff_late_opt_cagr']:+.2f}%/yr at the late optimum\")"
        ),
        md("### 4b. The start-date sweep — the sharpest instability in the study\n\n"
           "The era hand-off above depends on a boundary the analyst picks. So does the "
           "*sample start*, and that one is usually invisible: it is set by whatever "
           "history the data vendor happens to hold. Re-run the entire headline from five "
           "different start dates, holding the split, the as-of date and every parameter "
           "fixed.\n\n"
           "> ⚠️ **Why this section exists.** The first draft of this study ran on a cache "
           "whose ^IRX began 2004-01-06 and reported the hand-off with the **opposite "
           "sign** (+7.07% vs +7.81% unlevered — a loss). A later cache refresh pushed the "
           "start back to 2003-06-04 and the conclusion flipped. Nothing about the world "
           "changed. Rather than silently restate the headline, the sensitivity is now "
           "measured and shipped."),
        code(
            "print(f\"{'start':>12}{'n':>7}{'opt':>7}{'Kelly':>8}{'CAGR@opt':>10}\"\n"
            "      f\"{'handoff':>10}{'unlev':>8}{'edge':>9}\")\n"
            "for i, s0 in enumerate(R['ss_starts']):\n"
            "    print(f\"{s0:>12}{R['ss_n'][i]:7d}{R['ss_opt'][i]:7.2f}{R['ss_kelly'][i]:8.2f}\"\n"
            "          f\"{R['ss_cagr_opt'][i]:+9.2f}%{R['ss_handoff'][i]:+9.2f}%\"\n"
            "          f\"{R['ss_unlev'][i]:+7.2f}%{R['ss_edge'][i]:+8.2f}%\")\n"
            "print()\n"
            "print(f\"realised optimum ranges {min(R['ss_opt']):.2f} -> {max(R['ss_opt']):.2f} \"\n"
            "      f\"and the hand-off edge {min(R['ss_edge']):+.2f}%/yr -> {max(R['ss_edge']):+.2f}%/yr\")\n"
            "print('-> the sign of the study\\'s central claim is a function of the left edge of the window')"
        ),
        md(f"Put the spread sweep next to this. Moving the financing assumption across its "
           f"entire 0-200 bps range moves the optimum by "
           f"{R['spread_opt'][0] - R['spread_opt'][4]:.2f}. Moving the *start date* by a few "
           f"years moves it by {max(R['ss_opt']) - min(R['ss_opt']):.2f}, and moves the "
           f"hand-off conclusion from +1.73%/yr to -4.32%/yr (and to +25.11%/yr if you "
           f"start after the crash). **The arbitrary choice nobody documents dominates the "
           f"assumption everybody argues about.**"),
        md("## 5. The tradable arm — ex-ante Kelly, one lag, log-growth test\n\n"
           "`μ/σ²` on the trailing 756 days through *t*, clipped to [1, 3], applied at "
           "*t+1*. Raced against fixed multiples over the same window.\n\n"
           "The test statistic is the HAC *t* on the daily **log**-return difference, "
           "because terminal wealth is a product and growth is additive in logs. The "
           "arithmetic-excess *t* is printed alongside precisely so the divergence is "
           "visible: it is mechanically inflated by the higher-vol arm and is **not** the "
           "test a compounding claim must pass.\n\n"
           "> 💡 **In plain words:** a levered arm has a bigger *average* daily return "
           "almost by definition. Only the log difference answers 'did it actually end up "
           "with more money, reliably?'"),
        code(
            "print(f\"window {R['kel_start']} -> {R['kel_end']}  n={R['kel_n']:,}\")\n"
            "print(f\"applied multiple: mean {R['kel_mean_lev']:.2f}, at cap {R['kel_at_cap']:.1f}% \"\n"
            "      f\"of days, at floor {R['kel_at_floor']:.1f}%\")\n"
            "print()\n"
            "print(f\"{'arm':<16}{'terminal':>10}{'CAGR':>9}{'exSharpe':>10}{'vol':>8}{'maxDD':>9}\")\n"
            "print(f\"{'ex-ante Kelly':<16}{R['kel_tw']:9.2f}x{R['kel_cagr']:+8.2f}%\"\n"
            "      f\"{R['kel_sharpe']:+10.3f}{R['kel_vol']:7.1f}%{R['kel_dd']:+8.1f}%\")\n"
            "print(f\"{'fixed L=1.00':<16}{R['l1_tw']:9.2f}x{R['l1_cagr']:+8.2f}%\"\n"
            "      f\"{R['l1_sharpe']:+10.3f}{R['l1_vol']:7.1f}%{R['l1_dd']:+8.1f}%\")\n"
            "print(f\"{'fixed L=2.00':<16}{R['l2_tw']:9.2f}x{R['l2_cagr']:+8.2f}%\"\n"
            "      f\"{R['l2_sharpe']:+10.3f}{'':7}{R['l2_dd']:+8.1f}%\")\n"
            "print()\n"
            "print(f\"vs L=1: log-growth advantage {R['adv_vs1']:+.2f}%/yr  HAC t={R['t_vs1']:+.2f}  \"\n"
            "      f\"95% CI [{R['ci_vs1_lo']:+.2f}%, {R['ci_vs1_hi']:+.2f}%]  \"\n"
            "      f\"(arithmetic-excess t={R['t_arith_vs1']:+.2f} <- inflated, not the test)\")\n"
            "print(f\"vs L=2: log-growth advantage {R['adv_vs2']:+.2f}%/yr  HAC t={R['t_vs2']:+.2f}\")"
        ),
        md("### 5a. The one place a multiple *can* move Sharpe\n\n"
           "Constant leverage cannot touch Sharpe — that is the identity above. But this "
           "arm is *time-varying*, so it can, and it does: +0.030. That is exactly the "
           "number a reader would seize on, so it gets a paired block-bootstrap CI rather "
           "than a shrug."),
        code(
            "print(f\"ex-ante Kelly {R['kel_sharpe']:+.3f}  vs  fixed 1x {R['l1_sharpe']:+.3f}\")\n"
            "print(f\"difference {R['sh_diff']:+.3f}   95% CI \"\n"
            "      f\"[{R['sh_ci_lo']:+.3f}, {R['sh_ci_hi']:+.3f}]   \"\n"
            "      f\"{R['sh_frac_pos']}% of draws positive\")\n"
            "print('-> the CI is ~14x the size of the point estimate: not distinguishable from zero')"
        ),
        md("### 5b. The advantage is a knob, not a signal\n\n"
           "Raise the cap and the advantage rises while the *t*-stat **falls**. A genuine "
           "signal scales its *t* with its magnitude; a leverage knob does not. Same story "
           "for the estimation window. No configuration reaches |*t*| = 2."),
        code(
            "print('cap sweep:')\n"
            "for c, a, t, dd in zip(R['caps'], R['cap_adv'], R['cap_t'], R['cap_dd']):\n"
            "    print(f'  cap {c:.1f}: advantage {a:+.2f}%/yr  t={t:+.2f}  maxDD {dd:+.1f}%')\n"
            "print('\\nestimation-window sweep:')\n"
            "for w, a, t in zip(R['wins'], R['win_adv'], R['win_t']):\n"
            "    print(f'  {w:5d}d: advantage {a:+.2f}%/yr  t={t:+.2f}')\n"
            "print(f\"\\nbest |t| anywhere in the design space: {max(R['cap_t'] + R['win_t']):.2f}  \"\n"
            "      f\"(desk bar for Real: |t| >= 2)\")"
        ),
        md("## 6. PROXY sweeps — testing the assumptions rather than trusting them"),
        code(
            "print('financing spread over bills (bps/yr):')\n"
            "for s, o, c, sh in zip(R['spreads'], R['spread_opt'], R['spread_cagr'], R['spread_sharpe']):\n"
            "    print(f'  {s:4d}: optimum {o:.2f}x  CAGR@opt {c:+.2f}%  exSharpe@opt {sh:+.3f} '\n"
            "          f\"(unlevered {R['curve_sharpe'][0]:+.3f})\")\n"
            "print('\\none-way reset cost (bps):')\n"
            "for c, o, g in zip(R['costs'], R['cost_opt'], R['cost_cagr']):\n"
            "    print(f'  {c:4.1f}: optimum {o:.2f}x  CAGR@opt {g:+.2f}%')\n"
            "print('\\n-> the borrow bites, the reset does not; and at EVERY assumption the')\n"
            "print('   excess Sharpe at the optimum sits below the unlevered Sharpe.')"
        ),
        md("## 7. Live synthetic control — the machinery is unbiased\n\n"
           "**Synthetic, not the real tape.** An i.i.d. Student-*t* world with a *planted* "
           "growth-optimal leverage of 2.0, and a null version earning exactly cash. The "
           "sweep must recover 2.0 on the first and collapse to the floor on the second. "
           "Conditional consistency (argmax vs the tape's own in-sample `μ/σ²`) is checked "
           "on a single tape, where it is near-exact.\n\n"
           "The per-seed scatter is not noise to be apologised for — it is the study's "
           "result, reproduced in a laboratory where the DGP is *known and stationary*."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from optimal_leverage import data, strategy as st\n"
            "GRID = np.round(np.arange(0.0, 3.0001, 0.25), 4)\n"
            "for tag, ss in [('planted Kelly = 2.0', 1.0), ('null (zero excess drift)', 0.0)]:\n"
            "    opts, kels = [], []\n"
            "    for s in range(8):\n"
            "        d = st.synthetic_detect(data.synthetic_daily(signal_strength=ss, seed=944+s)[0], grid=GRID)\n"
            "        opts.append(d['opt_lev']); kels.append(d['kelly'])\n"
            "    print(f\"{tag:26s} argmax mean {np.mean(opts):.2f} (sd {np.std(opts, ddof=1):.2f})  \"\n"
            "          f\"Kelly mean {np.mean(kels):+.2f}\")\n"
            "    print(f\"{'':26s} per-seed argmax {sorted(opts)}\")\n"
            "lg = st.prepare_synth(data.synthetic_daily(signal_strength=1.0, seed=944)[0])\n"
            "fine = st.realised_optimum(lg, grid=np.round(np.arange(0.0, 4.0001, 0.05), 4),\n"
            "                           spread_bps=0.0, cost_bps=0.0)\n"
            "print(f\"\\nconditional consistency on one 40-year tape: argmax {fine:.2f} vs \"\n"
            "      f\"in-sample Kelly {st.kelly_from_legs(lg):.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The growth curve is real, concave and peaks at "
           f"**{R['opt']:.2f}×** against a Kelly of {R['kelly']:.2f}× — directionally exactly "
           f"what theory predicts, and positive in both eras and at every assumption. But "
           f"the *location* of the peak is unidentified: block-bootstrap CI "
           f"**[{R['boot_ci_lo']:.2f}, {R['boot_ci_hi']:.2f}]** (the whole grid, sd "
           f"{R['boot_sd']:.2f}); the five-year hindsight optimum sits at the floor in "
           f"{R['roll_at_floor']:.0f}% of windows and the cap in {R['roll_at_cap']:.0f}%; the "
           f"ex-ante Kelly arm beats 1× by {R['adv_vs1']:+.2f}%/yr at HAC *t* = "
           f"{R['t_vs1']:+.2f} with a CI spanning zero, and by only {R['adv_vs2']:+.2f}%/yr "
           f"(*t* = {R['t_vs2']:+.2f}) against an arbitrary fixed 2×. Best |*t*| anywhere in "
           f"the design space: {max(R['cap_t'] + R['win_t']):.2f}. Positive sign everywhere, "
           f"significance nowhere.\n"
           f"- **Tradability — Mirage.** Excess Sharpe is invariant in L by construction "
           f"({R['sharpe_gross_l1']:.4f} at 1× and 3× gross of financing) and falls to "
           f"{R['curve_sharpe'][4]:.3f} at the optimum once the spread is paid — nothing "
           f"risk-adjusted exists to bank. The growth is bought with a "
           f"{R['curve_dd'][4]:.1f}% drawdown, and the number you would have to know is not "
           f"merely noisy but sample-dependent: the identical hand-off test reads "
           f"{R['ss_edge'][0]:+.2f}%/yr or {R['ss_edge'][3]:+.2f}%/yr against not levering "
           f"at all depending only on where the window starts. The late optimum applied "
           f"in the early era did stay ahead ({R['handoff_late_in_early']:+.2f}%/yr vs "
           f"{R['handoff_early_unlev']:+.2f}%) — through a "
           f"{R['handoff_late_in_early_dd']:.1f}% drawdown.\n"
           f"- **Caveat, named and priced.** The window is gated by cached ^IRX history and "
           f"opens in 2003-06: it contains the GFC and 2022 but not 2000-2002. A sample "
           f"opening in 2000 would place the realised optimum lower still — section 4b "
           f"measures how much that boundary is worth rather than merely confessing it. "
           f"The whole result is one "
           f"realisation of one index that did not suffer a terminal decade — survivorship "
           f"in its macro form, and it belongs on the Signal axis."),
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
