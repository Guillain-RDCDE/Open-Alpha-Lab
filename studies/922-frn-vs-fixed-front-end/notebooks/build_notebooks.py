"""Generate the two narrative notebooks for Study 922 (Floating-Rate Front End).

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


# Frozen real-tape headline — mirror of docs/results.md. USFR/TFLO/BIL/SHY total-return
# closes + ^IRX (yield index), 2014-02-04 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2014-02-04", end="2026-06-30", n_days=3118, fp="4bfc2a85745c",
    # levels: total return, vol, max drawdown, excess-of-cash, excess Sharpe
    usfr_ret=1.91, usfr_vol=1.46, usfr_dd=-2.12, usfr_exc=0.029, usfr_sh=0.02,
    tflo_ret=1.95, tflo_vol=1.98, tflo_dd=-5.01, tflo_exc=0.067, tflo_sh=0.03,
    bil_ret=1.75, bil_vol=0.26, bil_dd=-0.24, bil_exc=-0.131, bil_sh=-0.56,
    shy_ret=1.48, shy_vol=1.47, shy_dd=-5.71, shy_exc=-0.406, shy_sh=-0.28,
    # pairwise race (annualised %, HAC t)
    usfr_bil=0.160, t_usfr_bil=0.68, tflo_bil=0.198, t_tflo_bil=0.96,
    usfr_shy=0.435, t_usfr_shy=0.96, bil_shy=0.275, t_bil_shy=0.72,
    usfr_tflo=-0.038, t_usfr_tflo=-0.13,
    # liquidity eras
    early_usfr_bil=0.169, early_t=0.23, late_usfr_bil=0.155, late_t=1.40,
    late_tflo_bil=0.143, late_tflo_t=2.08, late_usfr_shy=0.693, late_usfr_shy_t=1.24,
    # ^IRX regime cut
    reg_rise_n=626, reg_flat_n=2038, reg_fall_n=390,
    rise_usfr=2.65, rise_bil=2.12, rise_shy=-0.04, rise_gap=2.69, rise_gap_t=1.94,
    flat_usfr=1.55, flat_bil=1.49, flat_shy=1.73, flat_gap=-0.18, flat_gap_t=-0.35,
    fall_usfr=3.14, fall_bil=2.84, fall_shy=2.87, fall_gap=0.26, fall_gap_t=0.23,
    # the headline contrast
    contrast=2.43, contrast_t=1.37, rising_extra=2.87, rising_extra_t=1.92,
    falling_extra=0.45, falling_extra_t=0.37,
    contrast_bil=2.19, contrast_bil_t=1.33,
    sweep_positive=12, sweep_total=12, sweep_fire=0,
    sweep_lo=0.03, sweep_hi=6.41, sweep_best=2.69, sweep_best_t=1.94,
    # cycle windows
    zirp_usfr=0.73, zirp_bil=0.58, zirp_shy=1.11, zirp_gap=-0.38, zirp_gap_t=-0.80,
    hike_usfr=3.39, hike_bil=2.85, hike_shy=-1.07, hike_gap=4.46, hike_gap_t=1.99,
    hike_usfr_bil=0.54, hike_usfr_bil_t=2.05, hike_n=356,
    plat_usfr=4.83, plat_bil=5.24, plat_shy=5.50, plat_gap=-0.66, plat_gap_t=-0.40,
    cut_usfr=4.24, cut_bil=4.07, cut_shy=3.39, cut_gap=0.86, cut_gap_t=0.87,
    # drawdowns, liquid era
    usfr_dd_liq=-0.40, tflo_dd_liq=-0.16, bil_dd_liq=-0.21, shy_dd_liq=-5.71,
    usfr_vol_liq=0.60, shy_vol_liq=1.67,
    # bootstrap
    ci_usfr_bil_lo=-0.244, ci_usfr_bil_hi=0.548, ci_usfr_bil_neg=21.4,
    ci_usfr_shy_lo=-0.432, ci_usfr_shy_hi=1.321, ci_usfr_shy_neg=15.0,
    # costs, borrow, proxy
    cost1_1y=0.395, cost5_1y=0.235, cost5_3y=0.368,
    borrow25=0.185, borrow50=-0.065, borrow100=-0.565,
    bil_sh_360=1.836,
    # duration attribution
    hike_dirx=4.94, hike_gap_cum=6.45, hike_pred=9.13,
    cut_dirx=-1.24, cut_gap_cum=1.68, cut_pred=-2.29,
    # synthetic control
    syn_contrast=8.35, syn_t=14.2, syn_rise=5.43, syn_fall=-2.92,
    syn_null_mean=0.14, syn_null_sd=0.40, syn_null_fire=0,
)


HEADER = f"""# Study 922 — Floating-Rate Front End 🪙

**Through hikes and cuts, which end of the cash curve actually pays more?**

A **Treasury floating-rate note** pays a coupon that resets every week off the 13-week
bill auction, plus a small fixed spread. Because the coupon chases the rate, the price
barely moves — the note has almost no *duration*. The pitch writes itself: cash, with a
pickup, and none of the mark-to-market pain when the Fed hikes.

The counter-pitch is just as loud on the way down: a floater collects no term premium and
gets no capital gain when yields fall — the thing that makes a short *fixed* bond fund
worth owning into a cutting cycle.

We race **USFR** and **TFLO** (the two floaters) against **BIL** (1-3 month bills) and
**SHY** (1-3 year fixed) on daily **total-return** closes, {R['start']} → {R['end']}
({R['n_days']:,} days) — the first complete hike-plateau-cut cycle the Treasury FRN market
has ever lived through.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fp']}`); the
live cells run the fast synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Twelve years, four sleeves, one answer that keeps changing\n\n"
           "Held from 2014 to 2026, all four paid roughly the same — a bit under 2% a year, "
           "because most of that stretch was spent at zero. The floaters edged it. But the "
           "*averages* hide the story: which sleeve you wanted depended entirely on what "
           "rates were doing."),
        code(
            "R = dict(usfr=%r, tflo=%r, bil=%r, shy=%r,\n"
            "         usfr_dd=%r, tflo_dd=%r, bil_dd=%r, shy_dd=%r)\n"
            "for k in ('usfr', 'tflo', 'bil', 'shy'):\n"
            "    print('%%-5s total return %%+.2f%%%%/yr   worst drawdown %%+.2f%%%%'\n"
            "          %% (k.upper(), R[k], R[k + '_dd']))"
            % (R["usfr_ret"], R["tflo_ret"], R["bil_ret"], R["shy_ret"],
               R["usfr_dd"], R["tflo_dd"], R["bil_dd"], R["shy_dd"])
        ),
        md("## 2. 2022 was the floater's year — and only the floater's year\n\n"
           f"Split the tape by what the Fed was doing. Through the **hiking** window "
           f"(2022-03 → 2023-07, {R['hike_n']} trading days) the floaters made "
           f"**{R['hike_usfr']:+.2f}%/yr** while the 1-3 year fixed fund *lost* "
           f"**{R['hike_shy']:+.2f}%/yr** — a gap of **{R['hike_gap']:+.2f} pp/yr**. That is "
           f"not clever: it is arithmetic. A fund with ~1.85 years of duration loses about "
           f"1.85% of price for every 1 pp yields rise, and the front end rose "
           f"{R['hike_dirx']:.2f} pp.\n\n"
           f"Then look at the other windows. In the long **zero-rate** era the fixed fund won "
           f"({R['zirp_shy']:+.2f}% vs {R['zirp_usfr']:+.2f}%/yr). On the **plateau** it won "
           f"again ({R['plat_shy']:+.2f}% vs {R['plat_usfr']:+.2f}%). Two windows each way."),
        code(
            "rows = " + repr([
                ("ZIRP 2014-2021", R["zirp_usfr"], R["zirp_bil"], R["zirp_shy"]),
                ("hiking 2022-23", R["hike_usfr"], R["hike_bil"], R["hike_shy"]),
                ("plateau 2023-24", R["plat_usfr"], R["plat_bil"], R["plat_shy"]),
                ("cutting 2024-26", R["cut_usfr"], R["cut_bil"], R["cut_shy"]),
            ]) + "\n"
            "print(f\"{'window':16s} {'USFR':>8s} {'BIL':>8s} {'SHY':>8s}   winner\")\n"
            "for w, u, b, s in rows:\n"
            "    best = max((u, 'USFR'), (b, 'BIL'), (s, 'SHY'))[1]\n"
            "    print(f'{w:16s} {u:+7.2f}% {b:+7.2f}% {s:+7.2f}%   {best}')"
        ),
        md("## 3. The surprise: the cutting cycle never paid duration back\n\n"
           f"The textbook says the fixed fund gets its revenge when rates fall — its price "
           f"rises as yields drop. Duration predicted SHY should beat the floater by about "
           f"**{abs(R['cut_pred']):.1f}%** cumulatively over 2024-09 → 2026-06.\n\n"
           f"It didn't. The floater still won by **{R['cut_gap_cum']:+.2f}%**. Why? Because "
           f"the curve stayed flat-to-inverted: the 13-week rate fell "
           f"{abs(R['cut_dirx']):.2f} pp but the 1-3 year yield fell far less, and SHY was "
           f"starting from a *lower* yield than cash in the first place. You paid for "
           f"duration and the cheque has not arrived.\n\n"
           "> 🔬 **For the quants:** this is the time-varying term premium (Fama-Bliss, "
           "Cochrane-Piazzesi) in its most concrete form. The forward-implied compensation "
           "for two years of duration was negative for most of 2023-2025, so the 'insurance' "
           "leg had negative carry going in."),
        md("## 4. What is actually reliable here\n\n"
           f"Two things, and neither is a money machine.\n\n"
           f"**The pickup over bills is real but tiny.** USFR out-earned BIL by "
           f"**{R['usfr_bil']:+.3f}%/yr** and TFLO by **{R['tflo_bil']:+.3f}%/yr** — which is "
           f"just the floating note's spread over the bill rate, net of a 15 bp fee. The point "
           f"estimate is the same in 2014-2017 and 2018-2026. It is the size of a fee "
           f"decision, not an edge, and its *t*-statistic ({R['t_usfr_bil']:+.2f}) says so.\n\n"
           f"**The drawdown difference is enormous.** Since 2018, the floaters' worst loss was "
           f"**{R['usfr_dd_liq']:.2f}%** (USFR) and **{R['tflo_dd_liq']:.2f}%** (TFLO). SHY's "
           f"was **{R['shy_dd_liq']:.2f}%**, in October 2022. Same return, a fourteenth of the "
           f"pain — that is the real reason to own the floating end of the curve."),
        md("## 5. Why we won't call it Real\n\n"
           f"Everything above points the same way, and the effect is big. But "
           f"**twelve years contains exactly one hiking cycle**, and one macro event is one "
           f"observation however many trading days it spans. The single formal test — does the "
           f"direction of rates flip the ranking? — comes back at "
           f"**{R['contrast']:+.2f} pp/yr with *t* = {R['contrast_t']:+.2f}**, short of the "
           f"desk's |*t*| = 2 bar. The bootstrap interval on the floater-over-bills pickup, "
           f"[{R['ci_usfr_bil_lo']:+.3f}, {R['ci_usfr_bil_hi']:+.3f}], contains zero.\n\n"
           "> 🔬 **For the quants:** we lag every ^IRX-derived object exactly one day, use "
           "Newey-West errors and a 21-day block bootstrap precisely so the 626 'rising' days "
           "cannot masquerade as 626 independent draws."),
        md("## 6. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "Two toy worlds with the *same* rate cycle. In the first, the fixed leg has a real "
           "1.85-year duration, so rate direction must flip the ranking. In the second, the "
           "'fixed' leg is secretly a floater — rates still swing, but there is nothing to "
           "find. A trustworthy estimator shouts in the first and stays silent in the second."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from frn_front import data, strategy as st\n"
            "planted = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=922)[0])\n"
            "null    = st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=922)[0])\n"
            "print('fixed leg HAS duration : rate-direction contrast %+.2f pp/yr (t=%+.1f)'\n"
            "      % (planted['contrast'], planted['contrast_t']))\n"
            "print('fixed leg has NONE     : rate-direction contrast %+.2f pp/yr (t=%+.1f)'\n"
            "      % (null['contrast'], null['contrast_t']))\n"
            "print('(same rate cycle both times: %d rising days, %d falling days)'\n"
            "      % (null['n_rising'], null['n_falling']))\n"
            "import numpy as np\n"
            "nl = np.array([st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=922+s)[0])['contrast']\n"
            "               for s in range(8)])\n"
            "print('across 8 null worlds   : %+.2f pp/yr on average (spread %.2f) - noise, not signal'\n"
            "      % (nl.mean(), nl.std(ddof=1)))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The floating end really did out-pay the fixed end over the "
           f"cycle, and the ranking really does flip with the direction of rates — but the "
           f"whole margin was earned in one 17-month window, and **no test anywhere in the "
           f"study clears |*t*| = 2** (headline contrast {R['contrast']:+.2f} pp/yr, "
           f"*t* = {R['contrast_t']:+.2f}; 0 of 12 classifier settings). Textbook mechanism, "
           f"uncertified tape.\n"
           f"- **Tradability — Fragile.** ~15 bps/yr over bills is a fee-sized decision; "
           f"beating SHY requires knowing which way rates are going. What you *can* bank, for "
           f"free, is the risk profile: {R['usfr_dd_liq']:.1f}% worst drawdown against "
           f"{R['shy_dd_liq']:.1f}%. Choose your duration deliberately — don't expect the "
           f"floater to be alpha."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 922 — Floating-Rate Front End — the teardown\n\n"
           "The pairwise race with HAC *t*, the launch-liquidity era cut, the ^IRX regime "
           "split, the HAC-OLS rate-direction contrast and its classifier sweep, the cycle "
           "windows, the duration attribution, block-bootstrap CIs, and the borrow / cost / "
           "cash-proxy sweeps. Every real number is frozen from `docs/results.md` "
           "(fingerprint `%s`); the live cells are the synthetic control.\n\n"
           "**Conventions.** Daily total-return closes (`auto_adjust=True`); ^IRX is a "
           "price-only yield index and is never held. Exactly one execution lag: every "
           "^IRX-derived object (regime label, cash accrual) is formed through day *t* and "
           "applied at *t*+1. Sleeves are held, not traded, so cost is a round trip amortised "
           "over the holding horizon; read as a dollar-neutral pair, the short leg pays "
           "borrow. Survivorship: no cross-section — four named, still-listed funds."
           % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The four sleeves, and the excess-of-cash caveat\n\n"
           "Excess-of-cash uses the ^IRX/252 accrual **proxy**. Note how badly the *levels* "
           "behave: BIL's excess Sharpe is −0.56 on a 252 basis and **+1.84** on 360, because "
           "a few bps of convention dominate a 0.26%-vol series. Pairwise differences are "
           "invariant to the convention (cash cancels), which is why they are the headline."),
        code(
            "print(f\"{'fund':5s} {'ret':>8s} {'vol':>7s} {'maxDD':>8s} {'excess':>9s} {'exSharpe':>9s}\")\n"
            "for k in ('usfr', 'tflo', 'bil', 'shy'):\n"
            "    print(f\"{k.upper():5s} {R[k+'_ret']:+7.2f}% {R[k+'_vol']:6.2f}% {R[k+'_dd']:+7.2f}% \"\n"
            "          f\"{R[k+'_exc']:+8.3f}% {R[k+'_sh']:+9.2f}\")\n"
            "print(f\"\\ncash-proxy fragility: BIL excess Sharpe {R['bil_sh']:+.2f} on IRX/252 \"\n"
            "      f\"vs {R['bil_sh_360']:+.2f} on IRX/360 -> report differences, not levels\")"
        ),
        md("## 2. The pairwise race — no unconditional winner clears |*t*| = 2"),
        code(
            "pairs = [('USFR-BIL', R['usfr_bil'], R['t_usfr_bil']),\n"
            "         ('TFLO-BIL', R['tflo_bil'], R['t_tflo_bil']),\n"
            "         ('USFR-SHY', R['usfr_shy'], R['t_usfr_shy']),\n"
            "         ('BIL-SHY',  R['bil_shy'],  R['t_bil_shy']),\n"
            "         ('USFR-TFLO', R['usfr_tflo'], R['t_usfr_tflo'])]\n"
            "for p, d, t in pairs:\n"
            "    flag = '  <- |t|>=2' if abs(t) >= 2 else ''\n"
            "    print(f'{p:10s} {d:+.3f}%/yr   HAC t = {t:+.2f}{flag}')\n"
            "print(f\"\\nbootstrap (21d blocks): USFR-BIL 95% CI \"\n"
            "      f\"[{R['ci_usfr_bil_lo']:+.3f}, {R['ci_usfr_bil_hi']:+.3f}] \"\n"
            "      f\"({R['ci_usfr_bil_neg']:.1f}% of draws < 0)\")\n"
            "print(f\"                        USFR-SHY 95% CI \"\n"
            "      f\"[{R['ci_usfr_shy_lo']:+.3f}, {R['ci_usfr_shy_hi']:+.3f}] \"\n"
            "      f\"({R['ci_usfr_shy_neg']:.1f}% < 0)\")"
        ),
        md("## 3. Launch liquidity — a data-quality cut, not a regime cut\n\n"
           "USFR and TFLO listed in Feb-2014 and barely traded for years: USFR's quoted vol "
           "was 2.7% annualised in 2014 against 0.28% in 2025, and TFLO printed a +466 bp day "
           "next to a −446 bp day in Dec-2014. The *point estimate* of the pickup is stable "
           "across the split; only the noise shrinks. One cell (TFLO−BIL post-2018) touches "
           "*t* = 2 — one cell among many is not a rejection, and we do not treat it as one.\n\n"
           "> 💡 **In plain words:** the early prices are stale quotes on a fund nobody was "
           "trading, not losses anyone suffered."),
        code(
            "print(f\"2014-2017  USFR-BIL {R['early_usfr_bil']:+.3f}%/yr (t={R['early_t']:+.2f})\")\n"
            "print(f\"2018-2026  USFR-BIL {R['late_usfr_bil']:+.3f}%/yr (t={R['late_t']:+.2f})   \"\n"
            "      f\"TFLO-BIL {R['late_tflo_bil']:+.3f}%/yr (t={R['late_tflo_t']:+.2f})   \"\n"
            "      f\"USFR-SHY {R['late_usfr_shy']:+.3f}%/yr (t={R['late_usfr_shy_t']:+.2f})\")"
        ),
        md("## 4. Regime cut on the direction of ^IRX\n\n"
           "Label = sign of the 63-day change in the 13-week bill rate through *t*, dead band "
           "±0.25 pp, applied at *t*+1."),
        code(
            "rows = [('rising',  R['reg_rise_n'], R['rise_usfr'], R['rise_bil'], R['rise_shy'], R['rise_gap'], R['rise_gap_t']),\n"
            "        ('flat',    R['reg_flat_n'], R['flat_usfr'], R['flat_bil'], R['flat_shy'], R['flat_gap'], R['flat_gap_t']),\n"
            "        ('falling', R['reg_fall_n'], R['fall_usfr'], R['fall_bil'], R['fall_shy'], R['fall_gap'], R['fall_gap_t'])]\n"
            "print(f\"{'regime':8s} {'n':>5s} {'USFR':>7s} {'BIL':>7s} {'SHY':>7s} {'USFR-SHY':>10s} {'t':>7s}\")\n"
            "for r_, n, u, b, s, g, t in rows:\n"
            "    print(f'{r_:8s} {n:5d} {u:+6.2f}% {b:+6.2f}% {s:+6.2f}% {g:+9.2f}% {t:+7.2f}')"
        ),
        md("## 5. The headline test — one HAC-OLS regression, not a walk through sub-samples\n\n"
           "Regress the daily difference (annualised pp) on rising- and falling-rate dummies "
           "over the **whole** sample. The **contrast** = advantage when rates rise minus "
           "advantage when they fall. Right sign, large magnitude, *t* short of 2.\n\n"
           "> 💡 **In plain words:** the floater wins about 3 percentage points a year more "
           "when rates are climbing than when they are falling — but the sample cannot rule "
           "out that this is luck."),
        code(
            "print(f\"USFR-SHY: flat {R['flat_gap']:+.2f}%   rising extra {R['rising_extra']:+.2f} \"\n"
            "      f\"(t={R['rising_extra_t']:+.2f})   falling extra {R['falling_extra']:+.2f} \"\n"
            "      f\"(t={R['falling_extra_t']:+.2f})\")\n"
            "print(f\"  -> contrast {R['contrast']:+.2f} pp/yr (HAC t = {R['contrast_t']:+.2f})\")\n"
            "print(f\"BIL-SHY : contrast {R['contrast_bil']:+.2f} pp/yr (t = {R['contrast_bil_t']:+.2f})\")\n"
            "print(f\"\\nclassifier sweep ({R['sweep_total']} window/dead-band settings): \"\n"
            "      f\"positive in {R['sweep_positive']}/{R['sweep_total']}, range \"\n"
            "      f\"{R['sweep_lo']:+.2f} to {R['sweep_hi']:+.2f} pp/yr, |t|>=2 in \"\n"
            "      f\"{R['sweep_fire']}/{R['sweep_total']} (closest {R['sweep_best']:+.2f}, t={R['sweep_best_t']:+.2f})\")"
        ),
        md("## 6. Cycle windows and the duration attribution\n\n"
           "The windows are a declared **ASSUMPTION** (a hardcoded Fed calendar); the ^IRX cut "
           "above is the mechanical alternative and tells the same story. The attribution "
           "compares the realised cumulative USFR−SHY gap with the textbook *D × Δy*, where "
           "*D* ≈ 1.85 is SHY's published effective duration (an assumption, swept 1.60-2.10) "
           "and Δy is proxied by the **13-week** rate — deliberately crude, and the residual "
           "is exactly the curve."),
        code(
            "print(f\"{'window':16s} {'USFR':>7s} {'BIL':>7s} {'SHY':>7s} {'USFR-SHY':>10s} {'t':>7s}\")\n"
            "for w, u, b, s, g, t in [('ZIRP 2014-21', R['zirp_usfr'], R['zirp_bil'], R['zirp_shy'], R['zirp_gap'], R['zirp_gap_t']),\n"
            "                         ('hiking 22-23', R['hike_usfr'], R['hike_bil'], R['hike_shy'], R['hike_gap'], R['hike_gap_t']),\n"
            "                         ('plateau 23-24', R['plat_usfr'], R['plat_bil'], R['plat_shy'], R['plat_gap'], R['plat_gap_t']),\n"
            "                         ('cutting 24-26', R['cut_usfr'], R['cut_bil'], R['cut_shy'], R['cut_gap'], R['cut_gap_t'])]:\n"
            "    print(f'{w:16s} {u:+6.2f}% {b:+6.2f}% {s:+6.2f}% {g:+9.2f}% {t:+7.2f}')\n"
            "print(f\"\\nhiking : d(13w) {R['hike_dirx']:+.2f} pp -> predicted gap \"\n"
            "      f\"{R['hike_pred']:+.2f}%, realised {R['hike_gap_cum']:+.2f}% (curve inverted)\")\n"
            "print(f\"cutting: d(13w) {R['cut_dirx']:+.2f} pp -> predicted gap \"\n"
            "      f\"{R['cut_pred']:+.2f}%, realised {R['cut_gap_cum']:+.2f}%  <- duration never paid back\")\n"
            "print(f\"also: USFR-BIL in the hiking window {R['hike_usfr_bil']:+.2f}%/yr \"\n"
            "      f\"(t={R['hike_usfr_bil_t']:+.2f}) on {R['hike_n']} days - the reset-speed effect\")"
        ),
        md("## 7. Drawdowns, costs and borrow — what survives contact with reality\n\n"
           "Held sleeves pay a round trip once, so friction is amortised: even a punitive 5 bp "
           "spread leaves the USFR−SHY gap positive. Read instead as a dollar-neutral pair, "
           "the same difference is **dead by 50 bps of borrow**. The durable difference is the "
           "drawdown, and it costs nothing."),
        code(
            "print(f\"drawdowns 2018+: USFR {R['usfr_dd_liq']:+.2f}%  TFLO {R['tflo_dd_liq']:+.2f}%  \"\n"
            "      f\"BIL {R['bil_dd_liq']:+.2f}%  SHY {R['shy_dd_liq']:+.2f}%  \"\n"
            "      f\"(vol {R['usfr_vol_liq']:.2f}% vs {R['shy_vol_liq']:.2f}%)\")\n"
            "print(f\"held sleeve, USFR-SHY {R['usfr_shy']:+.3f}%/yr gross -> \"\n"
            "      f\"{R['cost1_1y']:+.3f}% at 1bp/1yr, {R['cost5_1y']:+.3f}% at 5bp/1yr, \"\n"
            "      f\"{R['cost5_3y']:+.3f}% at 5bp/3yr\")\n"
            "print(f\"long/short pair       : {R['borrow25']:+.3f}% at 25bp borrow, \"\n"
            "      f\"{R['borrow50']:+.3f}% at 50bp, {R['borrow100']:+.3f}% at 100bp\")"
        ),
        md("## 8. Live synthetic control — the machinery is unbiased\n\n"
           "Same rate cycle in both worlds; only the fixed leg's duration changes. Planted "
           "(D = 1.85): the contrast must be large and positive, with the floater ahead when "
           "rates rise and behind when they fall. Null (D = 0): the classifier is just as "
           "busy, but there is nothing to find. This proves the real-tape *t* = 1.37 is a "
           "sample-size fact, not a broken harness."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from frn_front import data, strategy as st\n"
            "pl = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=922)[0])\n"
            "print(f\"planted (D=1.85): contrast {pl['contrast']:+.2f} pp/yr (t={pl['contrast_t']:+.1f}); \"\n"
            "      f\"rising {pl['rising_extra']:+.2f}, falling {pl['falling_extra']:+.2f}\")\n"
            "nl = np.array([st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=922+s)[0])['contrast'] for s in range(8)])\n"
            "tn = np.array([st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=922+s)[0])['contrast_t'] for s in range(8)])\n"
            "print(f\"null x8 (D=0)   : contrast mean {nl.mean():+.2f} (sd {nl.std(ddof=1):.2f}), \"\n"
            "      f\"|t|>=2 in {(abs(tn)>=2).sum()}/8\")\n"
            "half = st.synthetic_detect(data.synthetic_panel(signal_strength=0.5, seed=922)[0])\n"
            "print(f\"half duration   : contrast {half['contrast']:+.2f} pp/yr -> the estimator scales with the planted effect\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** Correct sign everywhere and economically large — floaters "
           f"beat 1-3y fixed by {R['hike_gap']:+.2f} pp/yr through the hikes, lose by "
           f"{abs(R['zirp_gap']):.2f}-{abs(R['plat_gap']):.2f} pp/yr when rates sit still, and "
           f"the rising-minus-falling contrast is positive in "
           f"{R['sweep_positive']}/{R['sweep_total']} classifier settings. But nothing is "
           f"robust at |*t*| = 2: headline contrast {R['contrast']:+.2f} pp/yr "
           f"(*t* = {R['contrast_t']:+.2f}), unconditional pairs *t* = {R['t_usfr_bil']:+.2f} "
           f"to {R['t_usfr_shy']:+.2f}, all bootstrap CIs straddling zero, and the contrast "
           f"clears 2 in {R['sweep_fire']}/{R['sweep_total']} classifier settings. The ranking "
           f"flips with the regime and that flip *is* the finding — but no regime earns a "
           f"stamp of its own, so this is Weak (a mechanism the tape cannot certify), not "
           f"Mixed (a verdict that splits into stamps).\n"
           f"- **Tradability — Fragile.** The floater-over-bills pickup "
           f"({R['usfr_bil']:+.3f}%/yr) is stable and cost-proof but fee-sized; the "
           f"floater-over-SHY choice is a rate call; the pair dies at "
           f"{R['borrow50']:+.3f}%/yr on 50 bps of borrow. The bankable part is risk, not "
           f"return: {R['usfr_dd_liq']:.1f}% worst drawdown against {R['shy_dd_liq']:.1f}% for "
           f"the same total return.\n"
           f"- **Non-tape inputs**, all declared and swept: the ^IRX cash-accrual convention "
           f"(252 vs 360), the hardcoded Fed-cycle calendar, the regime window and dead band, "
           f"and SHY's assumed 1.85-year effective duration."),
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
