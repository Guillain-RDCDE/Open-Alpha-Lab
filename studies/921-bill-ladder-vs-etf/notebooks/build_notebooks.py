"""Generate the two narrative notebooks for Study 921 (Bill Ladder vs ETF).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the only live cells run the fast
synthetic control, so execution is quick and network-free.
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


# Frozen real-tape headline — mirror of docs/results.md. A 13-rung, 91-day held-to-maturity
# bill ladder priced off ^IRX, raced against BIL / SGOV / SHV total return,
# 2007-05-31 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2007-05-31", end="2026-06-30", n_days=4799, n_rolls=992, fp="1e3d76fa1bfa",
    # the race
    bil_cagr_l=1.4928, bil_cagr_e=1.3616, bil_gap=12.83, bil_t=2.75, bil_tnaive=1.15,
    sgov_start="2020-06-02", sgov_n=1527, sgov_cagr_l=2.9855, sgov_cagr_e=2.9571,
    sgov_gap=2.76, sgov_t=0.62,
    shv_n=4894, shv_cagr_l=1.5639, shv_cagr_e=1.5801, shv_gap=-1.65, shv_t=-0.34,
    vol_ladder=0.1464, vol_bil=0.4885,
    # fee attribution (bps/yr)
    bil_er=13.54, bil_gross=149.7, bil_resid=-0.4,
    sgov_er=9.00, sgov_gross=304.7, sgov_resid=-6.2,
    shv_er=15.00, shv_gross=173.0, shv_resid=-16.6,
    # bootstrap
    ci_lo=5.32, ci_hi=20.46, ci_neg=0.05,
    # eras
    era1_n=2163, era1_rate=0.49, era1_gap=10.46, era1_t=1.03,
    era2_n=1510, era2_rate=0.93, era2_gap=14.17, era2_t=3.23,
    era3_n=1126, era3_rate=3.97, era3_gap=15.61, era3_t=2.04,
    # rate-level cut
    zr_n=2785, zr_rate=0.15, zr_gap=12.64, zr_t=1.94,
    nr_n=2014, nr_rate=3.23, nr_gap=13.08, nr_t=1.68,
    # conventions & construction
    raw_gap=9.44, raw_t=2.02,
    rung4_gap=12.79, rung26_gap=12.82,
    # frictions
    c1_gap=8.83, c1_t=1.89, c2_gap=4.83, c2_t=1.03, c3_gap=0.83, c3_t=0.18,
    c5_gap=-7.17, c5_t=-1.53, c10_gap=-27.16, c10_t=-5.81,
    i1_gap=11.21, i1_t=2.40, i2_gap=9.59, i2_t=2.05, i3_gap=7.98, i3_t=1.71,
    i5_gap=4.74, i5_t=1.01,
    # conversion arithmetic
    conv_high=8.7, conv_low=1.1,
    # synthetic control
    syn_planted=13.35, syn_planted_sd=0.53, syn_null=-0.16, syn_null_sd=0.53, syn_fire=0,
    # --- inference audit: the bounce, the disclosed knobs, the knob-free arbiter ---
    acf1=-0.366,
    hac_scan=((0, 1.15), (1, 1.44), (2, 1.73), (5, 2.21), (9, 2.75),
              (21, 3.45), (63, 5.54), (252, 9.26)),
    boot_blocks=((5, 1.15, 24.07, 1.80), (10, 3.74, 22.19, 0.10),
                 (21, 5.32, 20.46, 0.05), (63, 8.37, 17.25, 0.00)),
    # non-overlapping period sums: (label, n_periods, mean bps/period, t)
    nov=(("weekly", 997, 0.246, 2.18), ("monthly", 230, 1.064, 3.27),
         ("quarterly", 77, 3.179, 3.54)),
    nov_raw=(("weekly", 1.60), ("monthly", 2.41), ("quarterly", 2.60)),
    # knob-free monthly t inside each era
    era1_tm=1.32, era2_tm=4.02, era3_tm=2.96,
)


HEADER = f"""# Study 921 — Bill Ladder vs ETF 🪜

**Does running your own 3-month T-bill ladder beat the cash ETF that charges you to run one?**

A cash ETF holds Treasury bills and rolls them. You can hold Treasury bills and roll them,
at TreasuryDirect, for nothing. So the forum arithmetic says a home-made ladder must beat
BIL by its expense ratio — "free money for ten minutes a quarter".

We **simulate** the ladder: **13 rungs of 91-day bills**, one bought every seven days and
held to maturity, priced off **^IRX** (the 13-week bill discount quote, converted to a
bond-equivalent yield). Say that plainly — the ladder leg is *modelled*, and ^IRX is a
secondary-market quote standing in for the auction stop-out a real buyer would receive, so
the ladder is the arithmetic of a ladder rather than a track record. Only the funds are a
traded tape. We race it against **BIL**, **SGOV** and **SHV** total return over
{R['start']} → {R['end']} ({R['n_days']:,} days, {R['n_rolls']} rolls). Cash is the
numeraire, so nothing is excess of anything: the number is the **annualised gap in
basis points a year**. One execution lag — yesterday's quote prices today's purchase.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
live cells run the fast offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The claim, and why it should be boringly true\n\n"
           "Almost every idea on this desk is a story about crowds or risk premia, and almost "
           "every one dies. This one is different: it is arithmetic. If two portfolios hold "
           "the same Treasury bills and only one of them pays a manager, the other should win "
           "by exactly the manager's fee. Nothing to forecast, nothing to be right about.\n\n"
           "So the interesting questions are not *does it work* but **how big is it**, "
           "**does the tape actually show it**, and **what does it cost you to collect**."),
        md("## 2. The tape says yes — by thirteen basis points\n\n"
           "Nineteen years of daily data. The ladder wins, and the margin is small enough to "
           "fit in a rounding error on a stock chart."),
        code(
            "R = dict(bil_gap=%r, bil_t=%r, bil_cagr_l=%r, bil_cagr_e=%r,\n"
            "         bil_er=%r, bil_gross=%r, bil_resid=%r, ci_lo=%r, ci_hi=%r)\n"
            "print('home-made ladder : %%.4f%%%% a year' %% R['bil_cagr_l'])\n"
            "print('BIL (the ETF)    : %%.4f%%%% a year' %% R['bil_cagr_e'])\n"
            "print('gap              : %%+.2f basis points a year  (HAC t = %%+.2f)'\n"
            "      %% (R['bil_gap'], R['bil_t']))\n"
            "print('95%%%% bootstrap CI : [%%+.2f, %%+.2f] bps/yr -- comfortably above zero'\n"
            "      %% (R['ci_lo'], R['ci_hi']))"
            % (R["bil_gap"], R["bil_t"], R["bil_cagr_l"], R["bil_cagr_e"],
               R["bil_er"], R["bil_gross"], R["bil_resid"], R["ci_lo"], R["ci_hi"])
        ),
        md("## 3. And the gap is *exactly* the fee — not a penny more\n\n"
           f"Here is the cleanest number in the study. BIL charges **{R['bil_er']:.2f} bps** a "
           f"year. Add that fee back to what BIL actually paid its holders and you get its "
           f"return *before* fees: **{R['bil_gross']:.1f} bps a year**. The ladder earned "
           f"**{R['bil_cagr_l']*100:.1f} bps a year**.\n\n"
           f"The difference is **{R['bil_resid']:+.1f} basis points a year** — under half a "
           f"basis point, over nineteen years. There is no cleverness in the ladder, no "
           f"secret pickup, no skill. It is the fee, and only the fee.\n\n"
           "> 🔬 **For the quants** — that residual is the fee identity of Bogle (2014) and "
           "Fama-French (2010) measured in the one asset class where security selection has "
           "nowhere to hide, so the coefficient on expenses should be exactly −1. It is."),
        md("## 4. The tell: the gap does not care what rates are\n\n"
           f"Split the sample not by date but by the *level* of short rates. When bills yielded "
           f"an average of **{R['zr_rate']:.2f}%** the gap was **{R['zr_gap']:+.2f} bps**; when "
           f"they yielded **{R['nr_rate']:.2f}%** it was **{R['nr_gap']:+.2f} bps**. Flat.\n\n"
           "That is the signature of a fee. A fund takes its 13.5 basis points whether bills pay "
           "five basis points or five per cent. If the ladder were instead earning some clever "
           "carry, the gap would grow with the rate level. It doesn't budge."),
        md("## 5. Now the bad news — three basis points of friction eats all of it\n\n"
           f"A 13-rung weekly ladder buys **52 bills a year**. Charge even **1 bp** of "
           f"round-trip friction on each purchase and the gap falls to **{R['c1_gap']:+.2f}**; "
           f"at **3 bps** it is **{R['c3_gap']:+.2f}** — gone (*t* = {R['c3_t']:+.2f}).\n\n"
           f"Same story if your maturing bill sits in cash before the replacement settles. "
           f"Three idle days a roll and the gap is **{R['i3_gap']:+.2f}** (*t* = "
           f"{R['i3_t']:+.2f}).\n\n"
           "At TreasuryDirect with auto-reinvestment you pay neither. Through a broker's "
           "secondary-market bill desk you pay both, and the whole exercise is a wash."),
        md("## 6. And the easy way collects most of it anyway\n\n"
           f"Race the ladder against **SGOV** — the cheap modern competitor — instead of BIL, "
           f"and the gap collapses to **{R['sgov_gap']:+.2f} bps a year** (*t* = "
           f"{R['sgov_t']:+.2f}). Buying a cheaper fund captures nearly the whole edge, with "
           f"none of the 52 auctions.\n\n"
           f"One more warning. The ladder's measured volatility is **{R['vol_ladder']:.2f}%** "
           f"against BIL's **{R['vol_bil']:.2f}%**, which looks like the ladder is safer. It "
           f"isn't. A bill held to maturity is simply never marked to market — the calm is an "
           f"accounting convention. Same credit, same maturity, *less* liquid: if you need the "
           f"money on a Tuesday you sell a bill at whatever the desk quotes, where the ETF "
           f"holder just hits a bid."),
        md("## 7. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "We build a fake world with a fake short rate and a fake cash ETF that charges a "
           "*known* fee, then run the same ladder pipeline over it. It should recover the fee "
           "we planted, and find nothing at all when the fake ETF is free."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from bill_ladder import data, strategy as st\n"
            "paid, truth = data.synthetic_daily(signal_strength=1.0, seed=921)\n"
            "free, _     = data.synthetic_daily(signal_strength=0.0, seed=921)\n"
            "print('planted ETF fee     : %.2f bps/yr' % truth['fee_bps_effective'])\n"
            "print('ladder recovers     : %+.2f bps/yr  (should match)'\n"
            "      % st.synthetic_detect(paid)['gap_bps'])\n"
            "print('free-ETF null       : %+.2f bps/yr  (should be ~0)'\n"
            "      % st.synthetic_detect(free)['gap_bps'])"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The (simulated) ladder really does beat BIL, by "
           f"**{R['bil_gap']:+.2f} bps a year** — significant on a test with nothing to tune "
           f"(non-overlapping monthly sums, *t* = +3.27), with HAC ({R['bil_t']:+.2f}) and the "
           f"bootstrap [{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}] agreeing — the same sign in all "
           f"three eras and flat in the level of rates. The gross-of-fee residual of "
           f"**{R['bil_resid']:+.1f} bps** says plainly what it is: the expense ratio, recovered. "
           f"It is arithmetic that came out right, not an edge anyone discovered.\n"
           f"- **Tradability — Fragile.** Thirteen basis points is the ceiling, and it dies at "
           f"3 bps of friction per purchase or 3 idle days per roll. Against SGOV it is already "
           f"down to {R['sgov_gap']:+.2f} bps. Buying the cheaper fund gets you almost all of "
           f"the edge for one click instead of fifty-two — and keeps the liquidity the ladder "
           f"quietly takes away."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 921 — Bill Ladder vs ETF — the teardown\n\n"
           "The ladder construction and its one execution lag, the discount→bond-equivalent "
           "conversion, the total-return race against three cash funds, the Newey-West *t* on "
           "the daily difference plus the full **inference audit** (the bounce diagnostic, the "
           "disclosed HAC-bandwidth and bootstrap-block scans, and the knob-free non-overlapping "
           "test that actually carries the verdict), the gross-of-fee attribution, the era and "
           "rate-level cuts, two friction "
           "sweeps, and the live synthetic control. Every real number is frozen from "
           "`docs/results.md` (Fingerprint `%s`)." % R["fp"]),
        code("R = %r" % (R,)),
        md("## Construction\n\n"
           "13 rungs of a 91-day bill, one bought every 7 calendar days, each held to maturity. "
           "The ladder's daily accrual is the equal-weighted mean of the 13 live rungs' locked "
           "bond-equivalent yields, applied actual/365 over **calendar** days — so a "
           "Friday→Monday step pays three days, the same clock the ETF's total-return close "
           "runs on.\n\n"
           "**The lag is exactly one.** The ^IRX quote at the close of day *t* prices the bill "
           "bought at *t+1*. Nothing else looks forward.\n\n"
           "**The conversion is not cosmetic.** ^IRX is a bank-discount rate, actual/360:\n\n"
           "```\n"
           "P   = 1 - d * 91 / 360\n"
           "BEY = (1 - P) / P * 365 / 91\n"
           "```\n\n"
           f"worth **+{R['conv_high']:.1f} bps** at a 3.70% quote and **+{R['conv_low']:.1f} bps** "
           f"at 0.50% — the same order as the effect under test, which is why the raw-quote "
           f"variant is reported as a floor.\n\n"
           "> 💡 **In plain words** — a discount rate is quoted off the face value you get "
           "back, not off the smaller price you paid, and on a 360-day year. Both corrections "
           "push the real yield up. Skip them and you understate the ladder by most of a fee."),
        md("## The headline race — total return vs total return\n\n"
           "Cash is the numeraire; there is nothing to take excess of. The statistic is the "
           "annualised mean daily return difference in bps/yr."),
        code(
            "print(f\"BIL  n={R['n_days']:5d}  ladder {R['bil_cagr_l']:.4f}%  ETF {R['bil_cagr_e']:.4f}%  \"\n"
            "      f\"gap {R['bil_gap']:+6.2f} bps/yr  HAC t {R['bil_t']:+.2f}  (naive t {R['bil_tnaive']:+.2f})\")\n"
            "print(f\"SGOV n={R['sgov_n']:5d}  ladder {R['sgov_cagr_l']:.4f}%  ETF {R['sgov_cagr_e']:.4f}%  \"\n"
            "      f\"gap {R['sgov_gap']:+6.2f} bps/yr  HAC t {R['sgov_t']:+.2f}\")\n"
            "print(f\"SHV  n={R['shv_n']:5d}  ladder {R['shv_cagr_l']:.4f}%  ETF {R['shv_cagr_e']:.4f}%  \"\n"
            "      f\"gap {R['shv_gap']:+6.2f} bps/yr  HAC t {R['shv_t']:+.2f}   <- duration control\")\n"
            "print(f\"\\nvol: ladder {R['vol_ladder']:.4f}% vs BIL {R['vol_bil']:.4f}% \"\n"
            "      f\"-- amortised cost, NOT less risk; no Sharpe race is quoted\")"
        ),
        md("## The inference audit — because HAC *helps* us here\n\n"
           f"Naive *t* = {R['bil_tnaive']:+.2f}, HAC *t* = {R['bil_t']:+.2f}. HAC usually *deflates* "
           "a *t*; here it more than doubles it. A correction that moves the result the author's "
           "way has to be audited, not asserted, so this section answers three questions in "
           "order: **why** the naive *t* is wrong, **how much** the tuning knobs move things, and "
           "**what the answer is with no knob at all**.\n\n"
           "**1. Why.** A bill ETF's daily close carries bid-offer bounce (Roll 1984), so the "
           "ladder-minus-ETF difference is a first difference of a stationary pricing error plus "
           "a drift, hence **negatively** autocorrelated at lag 1. Negative autocovariance shrinks "
           "the variance of the sample mean, so the i.i.d. standard error is too *large*."),
        code(
            "print(f\"lag-1 autocorrelation of the daily difference: {R['acf1']:+.3f}\")\n"
            "print('strongly negative = the Roll (1984) bounce signature. The naive SE is too big.')"
        ),
        md("**2. The knobs, disclosed.** HAC has a bandwidth; the block bootstrap has a block "
           "length. On this tape *both* push the same way — more lags, more significance — so the "
           "honest thing is to show the whole range and note where the headline sits in it."),
        code(
            "print('HAC t vs bandwidth:')\n"
            "for lags, t in R['hac_scan']:\n"
            "    tag = 'naive (i.i.d.)' if lags == 0 else f'HAC {lags:3d} lags'\n"
            "    mark = '   <- automatic rule, the headline' if lags == 9 else ''\n"
            "    print(f\"  {tag:16s} t {t:+.2f}{mark}\")\n"
            "print('\\nbootstrap CI vs block length:')\n"
            "for b, lo, hi, neg in R['boot_blocks']:\n"
            "    print(f\"  block {b:3d}d  95% CI [{lo:+6.2f}, {hi:+6.2f}]  share<0 {neg:.2f}%\")\n"
            "print('\\nThe automatic bandwidth sits near the BOTTOM of the kernel family, and the')\n"
            "print('shortest block gives the widest CI -- the headline is the conservative pick,')\n"
            "print('not the flattering one. But neither of these settles the question on its own.')"
        ),
        md("**3. The arbiter, which has no knob at all.** Sum the daily difference into "
           "**non-overlapping** calendar periods. Inside each period the bounce telescopes away "
           "(only the endpoints survive) while the accrual gap accumulates, so consecutive period "
           "sums are close to independent and an *ordinary* one-sample *t* is valid as it stands. "
           "There is no bandwidth and no block length to choose.\n\n"
           "This is the test the Real stamp actually rests on. HAC and the bootstrap merely concur."),
        code(
            "print('non-overlapping period sums -- ordinary t, nothing to tune:')\n"
            "for lab, n, mean, t in R['nov']:\n"
            "    print(f\"  {lab:10s} n={n:4d} periods  mean {mean:+.3f} bps/period  t {t:+.2f}\")\n"
            "print(f\"\\n  ... vs the naive daily t of {R['bil_tnaive']:+.2f} on the same data.\")\n"
            "print('  The knob-free test agrees with HAC, not with naive. Verdict evidence.')\n"
            "print('\\nsame test on the CONSERVATIVE raw-quote convention (no discount->BEY):')\n"
            "for lab, t in R['nov_raw']:\n"
            "    flag = '' if t >= 2.0 else '   <- honestly short of 2'\n"
            "    print(f\"  {lab:10s} t {t:+.2f}{flag}\")"
        ),
        md("> ⚠️ **What this does not rescue.** The raw-quote floor clears at monthly and above "
           "but falls to +1.60 weekly, and the ladder leg is still a *simulation* priced off a "
           "secondary-market quote. The inference is sound; the instrument is still modelled."),
        md("## Attribution — the gap *is* the expense ratio\n\n"
           "Add each fund's published expense ratio back to its net return to recover its gross "
           "return, and read the residual against the ladder. Expense ratios are a **PROXY** "
           "(sponsor stickers, not tape) and never enter a return calculation."),
        code(
            "for tag, cl, er, gr, res in [\n"
            "    ('BIL ', R['bil_cagr_l']*100, R['bil_er'], R['bil_gross'], R['bil_resid']),\n"
            "    ('SGOV', R['sgov_cagr_l']*100, R['sgov_er'], R['sgov_gross'], R['sgov_resid']),\n"
            "    ('SHV ', R['shv_cagr_l']*100, R['shv_er'], R['shv_gross'], R['shv_resid'])]:\n"
            "    print(f\"{tag}: ladder {cl:7.1f} bps/yr  vs ETF gross {gr:7.1f} (ER {er:5.2f})  \"\n"
            "          f\"-> residual {res:+6.1f} bps/yr\")\n"
            "print('\\nBIL residual -0.4 bps over 19 years: no curve pickup, no tenor bonus, no skill.')\n"
            "print('SGOV -6.2: its EFFECTIVE whole-period fee was ~3 bps, not the 9 bps sticker.')\n"
            "print('SHV -16.6: the duration control earning a curve pickup the ladder forgoes.')"
        ),
        md("> 💡 **In plain words** — if the ladder were doing anything other than dodging a fee, "
           "this residual would not be zero. It is zero for BIL, negative for the longer-maturity "
           "fund (which earns something the ladder can't), and negative for SGOV in the amount by "
           "which its sticker overstates what it actually charged."),
        md("## Era cut and rate-level cut\n\n"
           "The date slabs are cut out of the already-built difference series, so no slab loses a "
           "quarter to ladder warmup and the three partition the sample exactly. The rate-level "
           "cut is the sharper test: a fee is level-invariant, a carry is not."),
        code(
            "print('era                     gap      HAC t   knob-free monthly t')\n"
            "for lab, n, rate, gap, t, tm in [\n"
            "    ('2007-2015', R['era1_n'], R['era1_rate'], R['era1_gap'], R['era1_t'], R['era1_tm']),\n"
            "    ('2016-2021', R['era2_n'], R['era2_rate'], R['era2_gap'], R['era2_t'], R['era2_tm']),\n"
            "    ('2022-2026', R['era3_n'], R['era3_rate'], R['era3_gap'], R['era3_t'], R['era3_tm'])]:\n"
            "    flag = '' if tm >= 2.0 else '   <- sign only, NOT significant'\n"
            "    print(f\"{lab} n={n:5d} quote {rate:.2f}%  {gap:+6.2f}   {t:+.2f}    {tm:+.2f}{flag}\")\n"
            "print('\\n-> the SIGN is positive in all three, but only eras 2 and 3 are individually')\n"
            "print('   significant. \"Positive in all three eras\" is not three endorsements.')\n"
            "print()\n"
            "print(f\"quote <1%  n={R['zr_n']:5d}  mean {R['zr_rate']:.2f}%  \"\n"
            "      f\"gap {R['zr_gap']:+6.2f} (t={R['zr_t']:+.2f})\")\n"
            "print(f\"quote >=1% n={R['nr_n']:5d}  mean {R['nr_rate']:.2f}%  \"\n"
            "      f\"gap {R['nr_gap']:+6.2f} (t={R['nr_t']:+.2f})\")\n"
            "print('\\n-> a 22x change in the rate level moves the gap by 0.44 bps. That is a fee.')"
        ),
        md("## Assumption and construction sweeps"),
        code(
            "print(f\"discount->BEY (headline) : gap {R['bil_gap']:+6.2f} (t={R['bil_t']:+.2f})\")\n"
            "print(f\"raw quote (conservative) : gap {R['raw_gap']:+6.2f} (t={R['raw_t']:+.2f})  \"\n"
            "      f\"<- survives, but only just\")\n"
            "print(f\" 4 rungs (monthly roll)  : gap {R['rung4_gap']:+6.2f}\")\n"
            "print(f\"26 rungs (2x weekly)     : gap {R['rung26_gap']:+6.2f}  \"\n"
            "      f\"<- schedule moves it by 0.04 bps\")"
        ),
        md("## Friction sweeps — where it breaks\n\n"
           "52 rolls a year on 1/13 of NAV, so per-auction friction costs about **4x** its "
           "per-roll size in annual drag. Both frictions are PROXY / ASSUMPTION and swept "
           "rather than assumed. No short leg anywhere, so no borrow."),
        code(
            "for c, g, t in [(0, R['bil_gap'], R['bil_t']), (1, R['c1_gap'], R['c1_t']),\n"
            "                (2, R['c2_gap'], R['c2_t']), (3, R['c3_gap'], R['c3_t']),\n"
            "                (5, R['c5_gap'], R['c5_t']), (10, R['c10_gap'], R['c10_t'])]:\n"
            "    flag = '  <- edge gone' if (c > 0 and abs(t) < 2) else ('  <- now a LOSS' if g < 0 else '')\n"
            "    print(f\"{c:2d} bps/auction : gap {g:+7.2f} (t={t:+.2f}){flag}\")\n"
            "print()\n"
            "for d, g, t in [(0, R['bil_gap'], R['bil_t']), (1, R['i1_gap'], R['i1_t']),\n"
            "                (2, R['i2_gap'], R['i2_t']), (3, R['i3_gap'], R['i3_t']),\n"
            "                (5, R['i5_gap'], R['i5_t'])]:\n"
            "    print(f\"{d:2d} idle days   : gap {g:+7.2f} (t={t:+.2f})\")"
        ),
        md("## Live synthetic control — the machinery is unbiased\n\n"
           "An Ornstein-Uhlenbeck short rate, the matching 13-week discount quote, and a cash "
           "ETF that accrues that rate minus a *known* fee plus bid-offer bounce. The pipeline "
           "must recover the planted fee and must find nothing when the fee is zero. This "
           "proves the harness neither invents nor eats basis points; it never supports the "
           "real-tape stamp."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from bill_ladder import data, strategy as st\n"
            "pl = np.array([st.synthetic_detect(\n"
            "    data.synthetic_daily(signal_strength=1.0, seed=921+s)[0])['gap_bps'] for s in range(8)])\n"
            "nl = np.array([st.synthetic_detect(\n"
            "    data.synthetic_daily(signal_strength=0.0, seed=921+s)[0])['gap_bps'] for s in range(8)])\n"
            "print(f\"planted 13.50 bps fee x8: recovered {pl.mean():+.2f} (sd {pl.std(ddof=1):.2f})\")\n"
            "print(f\"free-ETF null        x8: recovered {nl.mean():+.2f} (sd {nl.std(ddof=1):.2f}), \"\n"
            "      f\"|gap|>=4 bps in {(np.abs(nl)>=4).sum()}/8\")\n"
            "half, tr = data.synthetic_daily(signal_strength=0.5, seed=921)\n"
            "print(f\"half fee ({tr['fee_bps_effective']:.2f} bps)  : recovered \"\n"
            "      f\"{st.synthetic_detect(half)['gap_bps']:+.2f} -- the response is linear\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** Gap **{R['bil_gap']:+.2f} bps/yr** vs BIL. The significance is "
           f"carried by the test with **no tuning knob** — non-overlapping period sums at *t* = "
           f"**+2.18 / +3.27 / +3.54** (weekly / monthly / quarterly) — with HAC "
           f"({R['bil_t']:+.2f}) and the block bootstrap "
           f"(**[{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]**, {R['ci_neg']:.2f}% of draws negative) "
           f"concurring, and the naive daily *t* of {R['bil_tnaive']:+.2f} understood to be too "
           f"small because the difference has a lag-1 autocorrelation of {R['acf1']:+.2f}. Flat in "
           f"the level of rates ({R['zr_gap']:+.2f} at 0.15% vs {R['nr_gap']:+.2f} at 3.23%), "
           f"invariant to the rung schedule, and surviving the conservative raw-quote convention "
           f"({R['raw_gap']:+.2f}, monthly *t* = +2.41). The gross-of-fee residual against BIL "
           f"is **{R['bil_resid']:+.1f} bps/yr**: the gap is the expense ratio, fully attributed. "
           f"The synthetic control recovers a planted fee ({R['syn_planted']:+.2f}, sd "
           f"{R['syn_planted_sd']:.2f}) and is silent on the null ({R['syn_null']:+.2f}, "
           f"{R['syn_fire']}/8). **Caveats named, not buried:** the ladder leg is *simulated* from "
           f"a rate index rather than traded and ^IRX is a secondary-market quote, not the auction "
           f"stop-out; era 1 (45% of the sample) is positive but **not individually significant** "
           f"(monthly *t* = {R['era1_tm']:+.2f}); and the raw-quote floor clears at monthly and "
           f"above but not at weekly (+1.60). An accounting identity confirmed, not a discovery.\n"
           f"- **Tradability — Fragile.** The headline charges **no friction to either leg**. The "
           f"edge is a fee, so it is capped by the fee — and it "
           f"dies at **3 bps** of per-auction friction ({R['c3_gap']:+.2f}, *t* = {R['c3_t']:+.2f}) "
           f"or **3 idle days** per roll ({R['i3_gap']:+.2f}, *t* = {R['i3_t']:+.2f}). Against "
           f"SGOV it is already only {R['sgov_gap']:+.2f} bps (*t* = {R['sgov_t']:+.2f}), so the "
           f"one-click substitute captures most of it. And the ladder's {R['vol_ladder']:.2f}% "
           f"volatility against BIL's {R['vol_bil']:.2f}% is amortised-cost accounting, not risk "
           f"reduction — the ladder is the same credit and the same tenor, held less liquidly.\n"
           f"- **Survivorship.** No cross-section, so no classic survivorship bias; but the three "
           f"funds raced are three that survived, and the expensive cash ETFs that closed are "
           f"absent. That omission biases *toward* the ladder, not against it."),
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
