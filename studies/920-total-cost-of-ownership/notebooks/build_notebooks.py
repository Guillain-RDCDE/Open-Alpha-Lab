"""Generate the two narrative notebooks for Study 920 (Total Cost of Ownership).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every **real-tape** number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the fast
offline synthetic control and the break-even arithmetic, and they are labelled as such. No
synthetic cell ever sits under a real-tape banner.
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
# Total-return closes (yfinance, auto_adjust=True), as-of 2026-06-30.
# Common window 2020-10-13 -> 2026-06-30 (fixed by QQQM's inception), fp c3aa747fad51.
# Tracking difference is always CHEAP MINUS LIQUID, in bp/yr.
# --------------------------------------------------------------------------- #
R = dict(
    asof="2026-06-30", fp_all="c10bd02337b7", fp_common="c3aa747fad51",
    cw_start="2020-10-13", cw_end="2026-06-30", cw_days=1434, cw_years=5,

    # stated expense ratios (PROSPECTUS ASSUMPTION, bp/yr)
    er_spy=9.45, er_ivv=3.0, er_voo=3.0, er_qqq=20.0, er_qqqm=15.0,

    # common-window tracking difference (bp/yr): stated gap, realised, sd, t,
    # bootstrap CI (_lo/_hi, TOO NARROW at n=5) and Student-t CI (_tlo/_thi, the honest one),
    # plus the count of positive years out of five.
    # ..._exact carries the unrounded TD so the notebooks' live break-even arithmetic
    # reproduces docs/results.md to the day instead of drifting by a rounding.
    ivv_gap=6.45, ivv_td=5.78, ivv_sd=4.43, ivv_t=2.92, ivv_lo=3.55, ivv_hi=7.90,
    ivv_tlo=0.2806, ivv_thi=11.2827, ivv_pos=5, ivv_exact=5.7816,
    voo_gap=6.45, voo_td=6.60, voo_sd=4.34, voo_t=3.40, voo_lo=3.46, voo_hi=9.33,
    voo_tlo=1.2141, voo_thi=11.9931, voo_pos=5, voo_exact=6.6036,
    pla_gap=0.00, pla_td=0.82, pla_sd=2.46, pla_t=0.75, pla_lo=-0.25, pla_hi=2.21,
    pla_tlo=-2.2369, pla_thi=3.8809, pla_pos=3, pla_exact=0.8220,
    qqm_gap=5.00, qqm_td=7.19, qqm_sd=3.05, qqm_t=5.27, qqm_lo=4.78, qqm_hi=9.48,
    qqm_tlo=3.4003, qqm_thi=10.9794, qqm_pos=5, qqm_exact=7.1898,
    qqm_years=(2.2, 8.8, 10.1, 8.4, 6.5),

    # the same quantity by the other two estimators, and the stub decomposition (bp of
    # total drift) that explains the disagreement: head + complete years + tail = total.
    ivv_cum=3.38, ivv_mon=3.90, ivv_head=5.7, ivv_yrs=28.9, ivv_tail=-15.4,
    voo_cum=8.07, voo_mon=9.04, voo_head=4.7, voo_yrs=33.0, voo_tail=8.2,
    pla_cum=4.69, pla_mon=5.14, pla_head=-1.0, pla_yrs=4.1, pla_tail=23.6,
    qqm_cum=4.48, qqm_mon=2.23, qqm_head=-5.0, qqm_yrs=35.9, qqm_tail=-5.5,

    # full histories (bp/yr) and the artefact autopsy
    full_ivv_td=2.24, full_ivv_t=0.78, full_ivv_trim=3.29, full_ivv_trim_t=1.41,
    full_ivv_worst_y=2016, full_ivv_worst=-40.9, full_ivv_n=25,
    full_voo_td=2.60, full_voo_t=0.69, full_voo_trim=5.48, full_voo_trim_t=4.65,
    full_voo_worst_y=2014, full_voo_worst=-47.3, full_voo_n=15,
    full_pla_td=0.86, full_pla_t=0.15, full_pla_worst_y=2014, full_pla_worst=-56.5,
    need_ivv=165, need_pla=2534,

    # era cut, full histories, split 2020-01-01
    era_ivv_early=1.14, era_ivv_early_t=0.30, era_ivv_late=5.78, era_ivv_late_t=2.92,
    era_voo_early=0.73, era_voo_early_t=0.12, era_voo_late=6.60, era_voo_late_t=3.40,
    era_qqm_early=5.52, era_qqm_early_t=1.68, era_qqm_late=7.42, era_qqm_late_t=7.93,

    # break-even (trading days) at a 0.5 / 1 / 2 / 5 bp round-trip spread differential
    be_ivv=(22, 44, 87, 218), be_voo=(19, 38, 76, 191),
    be_qqm=(18, 35, 70, 175), be_pla=(153, 307, 613, 1533),
    # pessimistic 1 bp break-even at each interval's low end: bootstrap (be_ci_*, too
    # optimistic) vs Student-t (be_t_*, the number the study quotes).
    be_ci_ivv=71, be_ci_voo=73, be_ci_qqm=53,
    be_t_ivv=898, be_t_voo=208, be_t_qqm=74,

    # empirical overlapping race, 1 bp extra round trip: (mean edge bp, win rate, HAC t).
    # Rows past 252 d are DESCRIPTIVE: the HAC bandwidth equals the holding period, so at
    # 1008 d over 425 overlapping windows the t is a bandwidth artefact, not evidence.
    emp_ivv={21: (-0.46, 0.432, -3.25), 63: (0.72, 0.620, 2.34), 126: (2.64, 0.777, 3.84),
             252: (6.76, 0.855, 4.07), 504: (16.65, 0.992, 4.31), 756: (30.33, 0.998, 7.33),
             1008: (39.60, 1.000, 35.83)},
    emp_voo={21: (-0.40, 0.433, -3.23), 63: (0.82, 0.631, 2.76), 126: (2.73, 0.788, 4.15),
             252: (6.86, 0.869, 4.12), 504: (16.94, 0.995, 4.35), 756: (30.66, 0.999, 7.70),
             1008: (40.78, 1.000, 36.41)},
    emp_qqm={21: (-0.51, 0.436, -3.05), 63: (0.73, 0.613, 2.68), 126: (2.65, 0.790, 5.19),
             252: (7.47, 0.921, 6.11), 504: (19.61, 0.987, 6.65), 756: (36.81, 0.994, 10.43),
             1008: (48.91, 0.993, 57.30)},
    emp_pla={21: (-0.94, 0.330, -10.54), 63: (-0.91, 0.350, -6.32), 126: (-0.91, 0.368, -4.37),
             252: (-0.90, 0.373, -3.55), 504: (-0.71, 0.420, -4.12), 756: (-0.67, 0.402, -3.48),
             1008: (0.18, 0.511, 0.35)},
    emp_first_pos=42, emp_first_sig=63, emp_first_pos_3bp=126, emp_first_sig_3bp=189,

    # the long/short harvest (common window), 1 bp one-way on four legs.
    # The placebo row is the control: no fee gap, yet the biggest harvest but one.
    ls_ivv_gross=0.95, ls_voo_gross=4.53, ls_qqm_gross=1.93, ls_pla_gross=3.58,
    ls_noise_ivv=71, ls_noise_voo=77, ls_noise_qqm=90, ls_noise_pla=52,
    ls_qqm_net5=-3.77, ls_qqm_net10=-8.77, ls_qqm_net25=-23.77,

    # excess-of-cash Sharpe race (BIL on BOTH legs), common window
    sh_spy=0.7802, sh_ivv=0.7878, sh_voo=0.7932, sh_qqq=0.7233, sh_qqqm=0.7289,
    sh_t_ivv=0.13, sh_t_voo=0.59, sh_t_qqm=0.11, sh_diff_pla=0.0054, sh_t_pla=0.59,
    arm_vol=16.8, arm_vol_q=22.5,

    # noise floor
    daily_sd_bp=5.0, mon_sd_qqm=10.0, ann_sd_qqm=3.0,

    # synthetic control
    syn=((0.0, -0.04, -0.02), (3.0, 3.41, 1.19), (6.0, 6.31, 3.75), (12.0, 13.63, 6.62)),
    syn_null_mean_t=-0.04, syn_null_sd_t=0.34, syn_null_fires=0, syn_null_n=8,
)


HEADER = f"""# Study 920 — Total Cost of Ownership 🧾

**Cheapest fee or tightest spread — at what holding period does each win?**

Two funds hold the same index. One is the mega-liquid original with the higher fee and the
tightest quote (SPY at {R['er_spy']} bp, QQQ at {R['er_qqq']} bp). The other is the cheap clone
with a slightly wider quote (IVV and VOO at {R['er_ivv']} bp, QQQM at {R['er_qqqm']} bp). The
cheap one wins a little every day you hold it and loses a little the day you buy it. So there
is a **break-even holding period**, and this study finds it.

Total cost = **expense ratio** (a prospectus number) + **realised tracking difference** (the
only piece actually on the tape) + **round-trip spread** (a quote-level number). We measure the
middle term on daily **total-return** closes and sweep the other two.

*Real-tape numbers below are the frozen headline (`docs/results.md`, common window
{R['cw_start']} → {R['cw_end']}, {R['cw_days']:,} days, fingerprint `{R['fp_common']}`,
as-of {R['asof']}). The live cells run only the offline synthetic control and the break-even
arithmetic, and say so.*
"""


def _emp_rows(d):
    return "\n".join(
        f"| {h} d | {v[0]:+.2f} bp | {v[1]:.0%} | {v[2]:+.2f} |" for h, v in sorted(d.items())
    )


# --------------------------------------------------------------------------- #
# 01 — for the curious
# --------------------------------------------------------------------------- #
def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),

        md("## 1. The sticker price is not the price\n\n"
           "Every fund publishes an expense ratio, and almost nobody checks whether that is what "
           "the fund actually cost them. It usually is not. A wrapper can lend its shares out and "
           "keep or hand back the revenue; it can hold a sliver of uninvested cash; it can be a "
           "1993-vintage *unit investment trust* that is legally forbidden from reinvesting "
           "dividends until the next quarterly payout. All of that lands in one measurable number: "
           "the **realised tracking difference** — how much more (or less) one fund actually "
           "delivered than its twin on the same index.\n\n"
           "> 🔬 **For the quants.** We measure it as the drift of `log(cheap / liquid)` on "
           "total-return closes, chained across complete calendar years. Total return is not "
           "optional: on price-only closes two funds with different dividend schedules would look "
           "like they had different tracking differences when they merely pay on different days."),

        md(f"## 2. What the tape says\n\n"
           f"Over the {R['cw_years']} complete years in which all these funds trade side by side "
           f"({R['cw_start']} → {R['cw_end']}):\n\n"
           f"| The cheap fund, over its pricier twin | Its fee advantage says | The tape delivered | *t* |\n"
           f"|---|--:|--:|--:|\n"
           f"| IVV over SPY | {R['ivv_gap']:+.2f} bp/yr | **{R['ivv_td']:+.2f} bp/yr** | {R['ivv_t']:+.2f} |\n"
           f"| VOO over SPY | {R['voo_gap']:+.2f} bp/yr | **{R['voo_td']:+.2f} bp/yr** | {R['voo_t']:+.2f} |\n"
           f"| QQQM over QQQ | {R['qqm_gap']:+.2f} bp/yr | **{R['qqm_td']:+.2f} bp/yr** | {R['qqm_t']:+.2f} |\n"
           f"| *VOO over IVV — both charge 3 bp* | {R['pla_gap']:+.2f} bp/yr | *{R['pla_td']:+.2f} bp/yr* | *{R['pla_t']:+.2f}* |\n\n"
           f"The advantage is real and it is roughly the size the prospectus implies. Each of the "
           f"three cheap funds beat its twin in **all five years** — QQQM's were "
           f"{', '.join(f'{v:+.1f}' for v in R['qqm_years'])} bp — while the same-fee pair managed "
           f"only three of five.\n\n"
           f"> ⚠️ **One thing this table cannot tell you: *why*.** SPY (1993) and QQQ (1999) are "
           f"*unit investment trusts*, a legal form that cannot reinvest dividends between "
           f"quarterly payouts. So they lose ground for two reasons at once — the higher fee *and* "
           f"the idle cash — and no pair on the tape separates them. What you are looking at is "
           f"what the cheap wrapper **delivered**, not what the fee **cost**."),

        md(f"## 3. The last row is the important one\n\n"
           f"VOO and IVV charge the **same** {R['er_voo']:.0f} basis points. If our ruler were "
           f"broken — if it manufactured a tracking difference out of noise — that pair would show "
           f"one. It does not: **{R['pla_td']:+.2f} bp/yr**, *t* = {R['pla_t']:+.2f}, and an "
           f"interval that comfortably includes zero.\n\n"
           f"That is what a placebo is for. The ruler fires where a fee gap exists and stays quiet "
           f"where none does, on the same tape, in the same window, with the same code."),

        md(f"## 4. So: how long must you hold?\n\n"
           f"The cheap fund saves you about **{R['qqm_td']:.0f} basis points a year**, forever. "
           f"Buying it costs you the extra spread, **once**. Divide one by the other and you get "
           f"the break-even. The cell below does exactly that arithmetic — it is a division, not a "
           f"backtest, so it runs live."),

        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from tco import strategy as st\n"
            "\n"
            "# Realised tracking differences, frozen from docs/results.md (real tape).\n"
            "td = {'IVV over SPY': %r, 'VOO over SPY': %r, 'QQQM over QQQ': %r,\n"
            "      'VOO over IVV (placebo)': %r}\n"
            "print('extra round-trip spread you pay for the cheap fund ->  break-even holding period')\n"
            "for name, t in td.items():\n"
            "    row = '  '.join(\n"
            "        ('never' if st.breakeven_days(t, s) == float('inf')\n"
            "         else '%%5.0f d' %% st.breakeven_days(t, s)) for s in (0.5, 1.0, 2.0, 5.0))\n"
            "    print('%%-24s (%%+.2f bp/yr):  %%s' %% (name, t, row))\n"
            "print('\\ncolumns: 0.5 bp   1 bp    2 bp    5 bp  of extra round-trip spread')"
            % (R["ivv_exact"], R["voo_exact"], R["qqm_exact"], R["pla_exact"])
        ),

        md(f"**About two months.** At a one basis-point wider round trip — a generous assumption "
           f"for funds this liquid — the cheap wrapper has repaid itself in "
           f"{R['be_qqm'][1]}–{R['be_ivv'][1]} trading days. Even at a punitive five basis points "
           f"it repays inside nine months. And the placebo pair, correctly, takes "
           f"{R['be_pla'][1]:,} days to repay a spread it has no advantage to repay it with.\n\n"
           f"> ⚠️ **Those are point estimates, and the tape is not precise.** Five years of data "
           f"pin a six basis-point number down only loosely. At the pessimistic end of an honest "
           f"interval the same one basis-point break-even stretches to **{R['be_t_qqm']} days for "
           f"QQQM/QQQ, {R['be_t_voo']} for VOO/SPY and {R['be_t_ivv']} for IVV/SPY** — three and a "
           f"half years for that last pair — and to *never* for the placebo. The direction is well "
           f"established; the *size* is not."),

        md(f"## 5. Does the tape agree with the arithmetic?\n\n"
           f"Dividing one number by another is easy. So we also walked every overlapping window on "
           f"the real tape — buy at tomorrow's close, hold *H* days, pay one extra basis point of "
           f"round trip — and asked when the cheap fund actually started winning. QQQM over QQQ:\n\n"
           f"| Hold | Average edge | Share of windows won | *t* |\n|---|--:|--:|--:|\n"
           + _emp_rows(R["emp_qqm"]) +
           f"\n\nPositive from **{R['emp_first_pos']} days**, statistically clear from "
           f"**{R['emp_first_sig']} days**, and by two years the cheap wrapper wins "
           f"{R['emp_qqm'][504][1]:.0%} of all windows. The arithmetic and the tape agree.\n\n"
           f"The placebo pair, over the same horizons, does **not** repay that one basis point: it "
           f"is still {R['emp_pla'][504][0]:+.2f} bp at two years and "
           f"{R['emp_pla'][756][0]:+.2f} bp at three, turning marginally positive "
           f"({R['emp_pla'][1008][0]:+.2f} bp) only at four."),

        md(f"## 6. Where this stops being worth anything\n\n"
           f"Two honest limits.\n\n"
           f"**It is not a trade.** If you try to *harvest* the gap — long the cheap fund, short "
           f"the expensive one — you capture only {R['ls_qqm_gross']:+.2f} bp/yr against "
           f"{R['ls_noise_qqm']} bp/yr of day-to-day tracking noise, and the borrow you pay on the "
           f"short leg buries it: {R['ls_qqm_net5']:+.2f} bp/yr at a five basis-point borrow, "
           f"{R['ls_qqm_net25']:+.2f} at twenty-five. This is a saving you *own*, not one you trade.\n\n"
           f"**It is small.** Six basis points a year is £6 per £10,000. Real, certain, and never "
           f"worth churning a position to capture. If you are already holding the expensive "
           f"wrapper in a taxable account with a large gain, the tax bill on switching dwarfs "
           f"decades of the saving."),

        md("## 7. The ruler, checked against a world we built\n\n"
           "The cell below is **synthetic** — a made-up pair of funds with a fee gap we planted "
           "ourselves. It is here only to show the measuring instrument is honest: it must recover "
           "a gap that is there, and report nothing when there is none."),

        code(
            "from tco import data\n"
            "for planted in (0.0, 6.0):\n"
            "    px, _ = data.synthetic_daily(gap_bp_yr=planted,\n"
            "                                 signal_strength=1.0 if planted else 0.0, seed=920)\n"
            "    d = st.synthetic_detect(px, n_boot=400)\n"
            "    print('planted %5.1f bp/yr -> recovered %+6.2f bp/yr  (t %+5.2f, 95%% CI [%+.2f, %+.2f])'\n"
            "          % (planted, d['td_ann_bp_yr'], d['t_annual'], d['ci_low'], d['ci_high']))"
        ),

        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The cheap wrapper's advantage is on the tape at "
           f"**{R['ivv_td']:+.2f} / {R['voo_td']:+.2f} / {R['qqm_td']:+.2f} bp/yr** "
           f"(*t* = {R['ivv_t']:+.2f} / {R['voo_t']:+.2f} / {R['qqm_t']:+.2f}), positive in **5/5 "
           f"years on each pair**, the size the prospectuses imply, and absent from the same-fee "
           f"placebo pair.\n"
           f"- **Tradability — Investable.** The break-even is about **two months** at a one "
           f"basis-point spread differential and under nine months at five. No timing, no "
           f"leverage, no forecast — just own the cheaper wrapper, which is why it is bankable "
           f"even though the tape measures it loosely. The prize is a basis point a month, so "
           f"never churn to get it, and never try to short the gap.\n"
           f"- **Three caveats we will not bury.** *(1)* The expensive fund in every winning pair "
           f"is a **unit investment trust**, so part of the gap is idle dividend cash rather than "
           f"the fee, and nothing here separates the two. *(2)* Measured over the *full* histories "
           f"rather than the common window, the S&P pairs miss significance "
           f"({R['full_ivv_t']:+.2f} and {R['full_voo_t']:+.2f}) — because the older public price "
           f"series contains adjustment errors worth **{R['full_pla_worst']:+.0f} bp in a single "
           f"year** between two funds that charge *identical* fees. *(3)* Even on the common "
           f"window, whether you see the gap at all depends on measuring it year-end to year-end: "
           f"the raw start-to-finish drift puts the same-fee placebo pair ({R['pla_cum']:+.2f} "
           f"bp/yr) **above** the genuine IVV/SPY pair ({R['ivv_cum']:+.2f}). The tape is dirtier "
           f"than the thing being measured; the quant notebook performs the autopsy."),
    ]
    nb["cells"] = cells
    return nb


# --------------------------------------------------------------------------- #
# 02 — for the quants
# --------------------------------------------------------------------------- #
def build_quants():
    nb = new_notebook()
    cells = [
        md(f"# Study 920 — Total Cost of Ownership — the teardown\n\n"
           f"The chained-period tracking-difference estimator, why the bootstrap runs on years "
           f"rather than months, the adjustment-artefact autopsy, the break-even curve and its "
           f"interval, the overlapping holding-period race with one execution lag, the borrow "
           f"sweep that kills the long/short version, and the live calibration check.\n\n"
           f"Every real-tape number is frozen from `docs/results.md` — common window "
           f"`{R['cw_start']} → {R['cw_end']}`, {R['cw_days']:,} days, fingerprint "
           f"`{R['fp_common']}`, as-of {R['asof']}. Total-return closes (`auto_adjust=True`); "
           f"tracking difference is always **cheap minus liquid, bp/yr**."),

        code("R = %r" % (R,)),

        md("## 1. The estimand and the estimator\n\n"
           "Two wrappers on one index. Model the log price ratio as a drift plus stationary noise:\n\n"
           "$$\\log\\frac{P^{cheap}_t}{P^{liquid}_t} = \\alpha t + \\varepsilon_t$$\n\n"
           "where $\\alpha$ is the annual tracking-difference advantage — fee gap, lending "
           "revenue, sampling error, cash drag, all of it — and $\\varepsilon_t$ is closing-print "
           "noise plus a slow level wobble. We report three estimators of $\\alpha$: the "
           "**cumulative** drift, the mean of **chained complete calendar years**, and the mean of "
           "**chained complete months** with a Newey-West *t*.\n\n"
           "Chaining matters. Tracking difference does not arrive smoothly — it lands in steps on "
           "distribution dates, which cluster at quarter ends. Measuring first-to-last *within* "
           "each period silently drops every period-boundary gap, i.e. exactly the sessions "
           "carrying the signal, and the three estimators then disagree with each other. Chaining "
           "removes *that* source of disagreement — but not the one that matters here, which is "
           "§2b.\n\n"
           "> 💡 **In plain words:** measure the gap from one year-end to the next, not from "
           "January to December — otherwise you throw away the New Year's gap, and that is where "
           "the money moves."),

        md("## 2. Why the bootstrap runs on years, not months\n\n"
           "$\\varepsilon_t$ is a **level** error, so chained differences of it are strongly "
           "*negatively* autocorrelated and telescope. On QQQ/QQQM the monthly tracking "
           "differences have a standard deviation near "
           f"{R['mon_sd_qqm']:.0f} bp while their twelve-month sums — which are just twelve of "
           f"them added up — have one near {R['ann_sd_qqm']:.0f} bp. Under independence the annual "
           "figure would be $\\sqrt{12} \\times 10 \\approx 35$ bp. A monthly-frequency interval "
           "therefore overstates the uncertainty by an order of magnitude, and a Bartlett kernel "
           "does not rescue it (the truncation throws away most of the negative mass). So the "
           "circular block bootstrap runs on complete years with two-year blocks.\n\n"
           "**And running it on years costs sample size: the common window has five.** A "
           "percentile block bootstrap on five points resamples an empirical distribution built "
           "from the very five numbers whose dispersion it is trying to price, with no allowance "
           "for $\\hat{\\sigma}$ being estimated from them. It comes out far too narrow. Every "
           "table below therefore carries a **Student-*t* interval beside the bootstrap one**, "
           "and every pessimistic claim in this study quotes the *t* interval."),

        md(f"## 2b. The three estimators disagree — the study's sharpest caveat\n\n"
           f"Chaining makes the estimators consistent *over the periods they share*. It cannot "
           f"reconcile them when they cover **different** spans, and they do: the annual estimator "
           f"keeps complete calendar years only, the cumulative one keeps the partial stubs at "
           f"both ends too. On the real common window that difference is not cosmetic:\n\n"
           f"| Pair | **Annual** | Cumulative | Monthly ×12 | opening stub | complete years | closing stub |\n"
           f"|---|--:|--:|--:|--:|--:|--:|\n"
           f"| IVV over SPY | **{R['ivv_td']:+.2f}** | {R['ivv_cum']:+.2f} | {R['ivv_mon']:+.2f} | "
           f"{R['ivv_head']:+.1f} | {R['ivv_yrs']:+.1f} | **{R['ivv_tail']:+.1f}** |\n"
           f"| VOO over SPY | **{R['voo_td']:+.2f}** | {R['voo_cum']:+.2f} | {R['voo_mon']:+.2f} | "
           f"{R['voo_head']:+.1f} | {R['voo_yrs']:+.1f} | {R['voo_tail']:+.1f} |\n"
           f"| VOO over IVV *(placebo)* | **{R['pla_td']:+.2f}** | **{R['pla_cum']:+.2f}** | "
           f"{R['pla_mon']:+.2f} | {R['pla_head']:+.1f} | {R['pla_yrs']:+.1f} | **{R['pla_tail']:+.1f}** |\n"
           f"| QQQM over QQQ | **{R['qqm_td']:+.2f}** | {R['qqm_cum']:+.2f} | {R['qqm_mon']:+.2f} | "
           f"{R['qqm_head']:+.1f} | {R['qqm_yrs']:+.1f} | {R['qqm_tail']:+.1f} |\n\n"
           f"The stubs are 2020 Q4 and 2026 H1, in bp of total drift. **Two funds charging the "
           f"same 3 bp cannot really drift {R['pla_tail']:+.1f} bp apart in six months** — that is "
           f"a level artefact in one closing print, four times the whole annual signal, and it is "
           f"what lifts the placebo's *cumulative* estimate to {R['pla_cum']:+.2f} bp/yr, **above "
           f"IVV-over-SPY's {R['ivv_cum']:+.2f}**.\n\n"
           f"So the headline is estimator-conditional, and honesty requires saying so plainly: a "
           f"reader who prefers the raw cumulative drift gets **no signal at all** on the IVV/SPY "
           f"pair. The defence is not that the annual estimator is larger. It is that the "
           f"complete-calendar-period rule was fixed ex ante by the as-of convention, and that the "
           f"placebo — whose true gap is zero by construction — demonstrates exactly the "
           f"contamination the rule removes. Two of three estimators (annual, and the overlapping "
           f"race in §7) leave the placebo quiet. The cumulative one does not."),

        md("## 3. The headline — common window, fixed by QQQM's inception\n\n"
           "The window is the only span on which all six wrappers trade. It was chosen by an "
           "inception date, not by a result — and the placebo pair is the guard against the "
           "suspicion that it was. Note that **all four pairs rest on the same five complete "
           "calendar years (2021–2025)**, including the S&P pairs whose own histories run to 15 "
           "and 25 years: *n* = 5 everywhere, so these are not four independent samples.\n\n"
           "The sign count is the one statement that survives without any interval at all: 5/5 "
           "positive years is a one-sided sign-test *p* of 1/32 = 0.031."),

        code(
            "hdr = ('pair', 'stated', 'realised', 'sd', 't', 'yrs+', 'boot lo', 'boot hi', 't-CI lo', 't-CI hi')\n"
            "print('%-24s %7s %9s %5s %6s %5s %8s %8s %8s %8s' % hdr)\n"
            "rows = [('IVV over SPY', 'ivv'), ('VOO over SPY', 'voo'),\n"
            "        ('QQQM over QQQ', 'qqm'), ('VOO over IVV [placebo]', 'pla')]\n"
            "for name, k in rows:\n"
            "    print('%-24s %+7.2f %+9.2f %5.2f %+6.2f %4d/5 %+8.2f %+8.2f %+8.2f %+8.2f'\n"
            "          % (name, R[k+'_gap'], R[k+'_td'], R[k+'_sd'], R[k+'_t'], R[k+'_pos'],\n"
            "             R[k+'_lo'], R[k+'_hi'], R[k+'_tlo'], R[k+'_thi']))\n"
            "print('\\nthe bootstrap interval is ~2-3x too narrow at n=5; the t interval is the honest one.')\n"
            "print('all three fee-gap pairs still clear zero on it -- IVV over SPY only just (%+.2f).'\n"
            "      % R['ivv_tlo'])\n"
            "print('\\nQQQM annual tracking differences (bp): %s  -> positive in 5/5 years'\n"
            "      % ', '.join('%+.1f' % v for v in R['qqm_years']))"
        ),

        md("## 4. The autopsy — why the full histories say less\n\n"
           "Run the same estimator over each pair's whole life and the S&P pairs collapse:\n\n"
           f"| Pair | Window | TD | *t* | Trimmed (*t*) | Worst single year |\n|---|---|--:|--:|--:|--:|\n"
           f"| IVV over SPY | {R['full_ivv_n']} yrs | {R['full_ivv_td']:+.2f} | {R['full_ivv_t']:+.2f} | "
           f"{R['full_ivv_trim']:+.2f} ({R['full_ivv_trim_t']:+.2f}) | {R['full_ivv_worst_y']}: **{R['full_ivv_worst']:+.1f} bp** |\n"
           f"| VOO over SPY | {R['full_voo_n']} yrs | {R['full_voo_td']:+.2f} | {R['full_voo_t']:+.2f} | "
           f"{R['full_voo_trim']:+.2f} (**{R['full_voo_trim_t']:+.2f}**) | {R['full_voo_worst_y']}: **{R['full_voo_worst']:+.1f} bp** |\n"
           f"| VOO over IVV | {R['full_voo_n']} yrs | {R['full_pla_td']:+.2f} | {R['full_pla_t']:+.2f} | — | "
           f"{R['full_pla_worst_y']}: **{R['full_pla_worst']:+.1f} bp** |\n\n"
           f"The diagnostic is the placebo row. Two funds charging the *same* three basis points "
           f"cannot truly differ by {R['full_pla_worst']:+.0f} bp in a year. Those are adjustment "
           f"artefacts in the public total-return series — a distribution timed into the wrong "
           f"session — and they are an order of magnitude larger than the effect being measured. "
           f"Drop the best and worst year and VOO-over-SPY lands on {R['full_voo_trim']:+.2f} bp/yr "
           f"at *t* = {R['full_voo_trim_t']:+.2f}, i.e. on its prospectus gap.\n\n"
           f"> 💡 **In plain words:** the fee gap did not appear in 2020. Our ability to *see* it did."),

        md("## 5. Power, and the era cut\n\n"
           "The cleanest way to state the same thing is a power calculation: at the observed annual "
           f"dispersion, reaching a one-sample |*t*| of 2 would take **{R['need_ivv']} years** on "
           f"the full IVV/SPY history and **{R['need_pla']:,} years** on the placebo pair. On the "
           f"common window the dispersion collapses to 3–4 bp and five years suffice."),

        code(
            "print('era cut, full histories, split 2020-01-01 (bp/yr, t)')\n"
            "for name, a, at, b, bt in [\n"
            "        ('IVV over SPY', R['era_ivv_early'], R['era_ivv_early_t'], R['era_ivv_late'], R['era_ivv_late_t']),\n"
            "        ('VOO over SPY', R['era_voo_early'], R['era_voo_early_t'], R['era_voo_late'], R['era_voo_late_t'])]:\n"
            "    print('  %-14s early %+5.2f (t %+.2f)   late %+5.2f (t %+.2f)' % (name, a, at, b, bt))\n"
            "print('  %-14s early %+5.2f (t %+.2f)   late %+5.2f (t %+.2f)   <- split 2023-01-01'\n"
            "      % ('QQQM over QQQ', R['era_qqm_early'], R['era_qqm_early_t'],\n"
            "         R['era_qqm_late'], R['era_qqm_late_t']))\n"
            "print('\\nyears of tape needed for |t|=2 at the observed dispersion:')\n"
            "print('  full IVV/SPY history: %d   placebo pair: %d' % (R['need_ivv'], R['need_pla']))"
        ),

        md("## 6. The break-even curve — the spread is a swept ASSUMPTION\n\n"
           "$H^\\ast = \\Delta\\text{spread} / \\alpha \\times 252$ trading days. No daily-close "
           "tape carries quotes, so $\\Delta\\text{spread}$ is swept end to end rather than "
           "assumed. The cell below runs the arithmetic live on the frozen real-tape $\\alpha$s "
           "(unrounded, so it reproduces `docs/results.md` to the day), showing the point estimate "
           "and **both** intervals' pessimistic ends so the gap between them is visible."),

        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from tco import strategy as st\n"
            "\n"
            "def _d(a, s):\n"
            "    v = st.breakeven_days(a, s)\n"
            "    return '%8s' % ('never' if v == float('inf') else '%.0f' % v)\n"
            "\n"
            "grid = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)\n"
            "print('%-24s %s' % ('break-even (trading days)', '  '.join('%6.1fbp' % s for s in grid)))\n"
            "for name, k in [('IVV over SPY', 'ivv'), ('VOO over SPY', 'voo'),\n"
            "                ('QQQM over QQQ', 'qqm'), ('VOO over IVV [placebo]', 'pla')]:\n"
            "    print('%-24s %s   <- point' % (name, ''.join(_d(R[k+'_exact'], s) for s in grid)))\n"
            "    print('%-24s %s   <- bootstrap low end (too optimistic)'\n"
            "          % ('', ''.join(_d(R[k+'_lo'], s) for s in grid)))\n"
            "    print('%-24s %s   <- Student-t low end (QUOTED)'\n"
            "          % ('', ''.join(_d(R[k+'_tlo'], s) for s in grid)))"
        ),

        md(f"At 1 bp the point estimates say {R['be_qqm'][1]}–{R['be_ivv'][1]} days. The bootstrap's "
           f"pessimistic end says {R['be_ci_qqm']}–{R['be_ci_voo']}; the Student-*t* end says "
           f"**{R['be_t_qqm']} / {R['be_t_voo']} / {R['be_t_ivv']} days**. That last row is the one "
           f"the study quotes, and it is the difference between *two and a half months* and *three "
           f"and a half years* for IVV/SPY. The direction of the effect is well established; its "
           f"magnitude is pinned down only to within a factor of about three."),

        md("## 7. The empirical race — one execution lag, overlapping windows\n\n"
           "The choice is made with information through the close of day *t*; both candidate "
           "positions are entered at the close of day *t+1*. The cheap wrapper is charged the "
           "extra round trip **once**, at entry — that is the entire execution-cost difference "
           "between the two choices, since one round trip is paid either way. Newey-West lags are "
           "set to the holding period (the overlap length). One basis point of extra round trip:\n\n"
           "| Hold | QQQM over QQQ | VOO over SPY | IVV over SPY | VOO over IVV *(placebo)* |\n"
           "|---|--:|--:|--:|--:|\n"
           + "\n".join(
               f"| {h} d | {R['emp_qqm'][h][0]:+.2f} ({R['emp_qqm'][h][1]:.0%}, {R['emp_qqm'][h][2]:+.2f}) "
               f"| {R['emp_voo'][h][0]:+.2f} ({R['emp_voo'][h][1]:.0%}, {R['emp_voo'][h][2]:+.2f}) "
               f"| {R['emp_ivv'][h][0]:+.2f} ({R['emp_ivv'][h][1]:.0%}, {R['emp_ivv'][h][2]:+.2f}) "
               f"| *{R['emp_pla'][h][0]:+.2f} ({R['emp_pla'][h][1]:.0%}, {R['emp_pla'][h][2]:+.2f})* |"
               for h in sorted(R["emp_qqm"])
           )
           + f"\n\nPositive from **{R['emp_first_pos']} days**, |*t*| ≥ 2 from "
           f"**{R['emp_first_sig']} days**, on all three real pairs — against an analytic "
           f"break-even of {R['be_qqm'][1]}–{R['be_ivv'][1]} days. At three basis points of extra "
           f"round trip those become {R['emp_first_pos_3bp']} and {R['emp_first_sig_3bp']} days. "
           f"This is the second estimator that leaves the placebo quiet, and the only one that "
           f"uses every window rather than year-end prints alone: the placebo is negative out to "
           f"**three** years ({R['emp_pla'][756][0]:+.2f} bp) and only marginally positive at four "
           f"({R['emp_pla'][1008][0]:+.2f} bp).\n\n"
           f"> ⚠️ **Read only the short rows as inference.** Newey-West lags are set to the "
           f"holding period, so the bandwidth-to-sample ratio is 0.05 at 63 days over ~1,370 "
           f"windows — fine — but 2.4 at 1,008 days over 425 windows, which contain barely one "
           f"independent draw. The *t* of {R['emp_qqm'][1008][2]:+.1f} in the bottom row is a "
           f"bandwidth artefact, not evidence, and the win rates have the same defect because "
           f"overlapping windows are not independent trials. **The 63-day row is the significance "
           f"claim; everything past 252 days is descriptive.**"),

        md("## 8. Why the long/short version is dead\n\n"
           "Long the cheap wrapper, short the liquid one, daily-rebalanced, 1 bp one-way on all "
           "four legs. This is the study's only short leg, so it is the only place borrow is paid — "
           "an ASSUMPTION with no tape behind it, hence swept."),

        code(
            "print('common window, long cheap / short liquid, 1 bp one-way x 4 legs')\n"
            "for name, g, v, tag in [('IVV / SPY', R['ls_ivv_gross'], R['ls_noise_ivv'], ''),\n"
            "                        ('VOO / SPY', R['ls_voo_gross'], R['ls_noise_voo'], ''),\n"
            "                        ('QQQM / QQQ', R['ls_qqm_gross'], R['ls_noise_qqm'], ''),\n"
            "                        ('VOO / IVV', R['ls_pla_gross'], R['ls_noise_pla'],\n"
            "                         '   <- PLACEBO: no fee gap to harvest')]:\n"
            "    print('  %-11s gross %+5.2f bp/yr   tracking noise %3d bp/yr   Sharpe ~ %.2f%s'\n"
            "          % (name, g, v, g / v, tag))\n"
            "print('\\nQQQM / QQQ net of borrow:  5 bp %+.2f   10 bp %+.2f   25 bp %+.2f  bp/yr'\n"
            "      % (R['ls_qqm_net5'], R['ls_qqm_net10'], R['ls_qqm_net25']))"
        ),

        md(f"**The placebo row is the verdict on this construction.** A pair with *no* fee gap "
           f"harvests {R['ls_pla_gross']:+.2f} bp/yr gross — more than IVV/SPY "
           f"({R['ls_ivv_gross']:+.2f}) and more than QQQM/QQQ ({R['ls_qqm_gross']:+.2f}). Daily "
           f"rebalancing of a mean-reverting print spread does not capture a fee; it captures "
           f"noise. It also gives most of the buy-and-hold gap back "
           f"({R['ls_qqm_gross']:+.2f} bp/yr traded against {R['qqm_td']:+.2f} held), and the "
           f"harvest sits inside {R['ls_noise_qqm']} bp/yr of tracking noise, so **any borrow above "
           f"about five basis points a year buries it**. The gap is a selection decision, never a "
           f"position.\n\n"
           f"> 💡 **In plain words:** you can keep this money by owning the right fund. You cannot "
           f"win it by betting on the difference."),

        md(f"## 9. The Sharpe race, and why it is the wrong instrument\n\n"
           f"Excess-of-cash (BIL) annualised Sharpe on the common window: SPY {R['sh_spy']:+.4f} vs "
           f"IVV {R['sh_ivv']:+.4f} vs VOO {R['sh_voo']:+.4f}; QQQ {R['sh_qqq']:+.4f} vs QQQM "
           f"{R['sh_qqqm']:+.4f} — both legs excess of the **same** cash series, so the cash leg "
           f"cancels in every difference. HAC *t* on the return differences: {R['sh_t_ivv']:+.2f}, "
           f"{R['sh_t_voo']:+.2f}, {R['sh_t_qqm']:+.2f}; the placebo pair's Sharpe difference "
           f"({R['sh_diff_pla']:+.4f}, *t* = {R['sh_t_pla']:+.2f}) is the same size as the real "
           f"pairs', which is the whole point.\n\n"
           f"A {R['qqm_td']:.0f} bp/yr fee gap inside an arm with {R['arm_vol']:.0f}–"
           f"{R['arm_vol_q']:.0f}% annualised volatility moves Sharpe in the third decimal. The "
           f"desk's usual instrument would have found nothing here — and would have been wrong. "
           f"The right test conditions on the fact that both arms hold *the same index*, which is "
           f"what differencing the log prices does."),

        md("## 10. Live calibration — the estimator is not merely non-zero, it is on the line\n\n"
           "Fully **synthetic** and offline: pairs built with a known planted gap, plus noise "
           "calibrated to the real tape (transient print error and a slow AR(1) level wobble). The "
           "estimator must land on the 45-degree line and stay at zero when nothing was planted."),

        code(
            "import numpy as np\n"
            "from tco import data\n"
            "panel, truths = data.synthetic_panel(gaps_bp_yr=(0.0, 3.0, 6.0, 12.0), seed=920)\n"
            "cal = st.panel_calibration(panel, truths, n_boot=300)\n"
            "print(cal[['planted_bp_yr', 'recovered_bp_yr', 't_annual']].round(2).to_string())\n"
            "nulls = np.array([\n"
            "    st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=920 + s)[0],\n"
            "                        n_boot=200)['t_annual'] for s in range(8)])\n"
            "print('\\nnull x8 seeds: mean t %+.2f (sd %.2f), |t|>=2 on %d/8'\n"
            "      % (nulls.mean(), nulls.std(ddof=1), int((abs(nulls) >= 2).sum())))"
        ),

        md(f"## Verdict\n\n"
           f"- **Signal — Real.** Realised tracking difference of "
           f"**{R['ivv_td']:+.2f} / {R['voo_td']:+.2f} / {R['qqm_td']:+.2f} bp/yr** at *t* = "
           f"**{R['ivv_t']:+.2f} / {R['voo_t']:+.2f} / {R['qqm_t']:+.2f}**, **positive in 5/5 years "
           f"on each pair** (sign-test *p* = 0.031), Student-*t* intervals clear of zero, within "
           f"~2 bp of the published fee gaps, and stable across both halves of QQQM's era cut. The "
           f"same-fee placebo prints {R['pla_td']:+.2f} bp/yr (*t* = {R['pla_t']:+.2f}, "
           f"{R['pla_pos']}/5 years) and stays negative for three years in the overlapping race. "
           f"The synthetic control is calibrated (0 → {R['syn'][0][1]:+.2f}, "
           f"6 → {R['syn'][2][1]:+.2f}, 12 → {R['syn'][3][1]:+.2f}) and fires "
           f"{R['syn_null_fires']}/{R['syn_null_n']} on the null — though it plants no mis-timed "
           f"distributions, so it cannot vouch for the estimator against the defect that actually "
           f"dominates the real series.\n"
           f"  **Named caveats, all load-bearing:** *(1)* the expensive leg is a **unit investment "
           f"trust in every pair that shows a gap**, so the measurement is fee + trust cash drag, "
           f"unseparated; *(2)* the result is **estimator-conditional** (§2b — the cumulative drift "
           f"puts the placebo at {R['pla_cum']:+.2f} above IVV/SPY's {R['ivv_cum']:+.2f}); *(3)* the "
           f"full pre-2020 histories miss significance because the public adjusted series carries "
           f"single-year artefacts of {R['full_pla_worst']:+.0f} bp between same-fee funds "
           f"(trimming restores {R['full_voo_trim']:+.2f} at *t* = {R['full_voo_trim_t']:+.2f}); "
           f"*(4)* all four pairs share the **same five years**, so these are not four independent "
           f"samples; *(5)* the funds are survivors — clones that closed never entered the sample.\n"
           f"- **Tradability — Investable.** Not because the tape pins the number down (it does "
           f"not) but because the act is a purchase decision with **no forecast, no timing, no "
           f"turnover and no capacity limit**, on a differential that is contractual and merely "
           f"confirmed here. Break-even **{R['be_qqm'][1]}–{R['be_ivv'][1]} trading days** at a "
           f"1 bp round-trip differential ({R['emp_first_pos']}–{R['emp_first_sig']} days "
           f"measured), under nine months at 5 bp — but **{R['be_t_qqm']}/{R['be_t_voo']}/"
           f"{R['be_t_ivv']} days at the honest interval's pessimistic end**. The boundary: worth "
           f"~6 bp/yr, so never churn a position for it, and the long/short expression is dead at "
           f"any borrow above ~5 bp/yr — where a same-fee placebo 'harvests' "
           f"{R['ls_pla_gross']:+.2f} bp/yr, more than two of the three real pairs.\n"
           f"- **Out of scope.** Intraday spreads (not on this tape), options-market depth, tax "
           f"lots, anything that separates the fee from the trust form, and which wrapper will be "
           f"cheapest *next* year — fees are a competitive variable, and Study 913 asks the "
           f"persistence question directly."),
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
