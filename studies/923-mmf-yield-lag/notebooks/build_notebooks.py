"""Generate the two narrative notebooks for Study 923 (The Cash Lag).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from
the frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the
fast three-world synthetic control, which is clearly labelled as synthetic and never
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


# Frozen real-tape headline — mirror of docs/results.md. BIL / USFR / SHV + ^IRX,
# daily total return, excess-of-BIL, 2014-02-04 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2014-02-04", end="2026-06-30", n_days=3118, fp="d78f3661ba09",
    # A1 — realised effective duration (years) and HAC t
    dur=dict(USFR=-0.001, SGOV=0.072, BIL=0.083, SHV=0.177),
    dur_t=dict(USFR=-0.04, SGOV=5.79, BIL=7.57, SHV=8.05),
    wam=dict(USFR=5, SGOV=25, BIL=45, SHV=110),
    dur_n=dict(USFR=3117, SGOV=1527, BIL=3117, SHV=3117),
    # A2 — distributed-lag pass-through (21d proxy)
    sum_beta=dict(SGOV=0.98, BIL=0.98, USFR=0.90, SHV=1.16),
    b0=dict(SGOV=-0.80, BIL=-1.16, USFR=0.27, SHV=-3.26),
    b0_t=dict(SGOV=-2.88, BIL=-9.99, USFR=0.18, SHV=-6.18),
    peak=dict(SGOV=35, BIL=21, USFR=56, SHV=21),
    centroid=dict(SGOV=26.7, BIL=23.1, USFR=40.4, SHV=22.7),
    # A4 — the measurement, era by era (duration years, HAC t)
    dur_era={
        "2014-2019": dict(USFR=(0.039, 0.32), BIL=(0.067, 2.82), SHV=(0.161, 6.64)),
        "2020-2026": dict(USFR=(-0.011, -0.44), BIL=(0.089, 7.59), SHV=(0.183, 7.19)),
    },
    # A3 — centroid lag by proxy window
    win_sweep={
        5: dict(SGOV=16.1, BIL=18.7, USFR=32.7, SHV=12.7),
        10: dict(SGOV=25.0, BIL=19.6, USFR=33.0, SHV=14.3),
        21: dict(SGOV=26.7, BIL=23.1, USFR=40.4, SHV=22.7),
        42: dict(SGOV=35.4, BIL=34.8, USFR=29.4, SHV=37.9),
    },
    # B1 — the arms (annualised bp excess of BIL, HAC t)
    sw_gross=-6.9, sw_gross_t=-0.29,
    sw_net=-115.3, sw_net_t=-4.39,
    placebo=-104.0, placebo_t=-4.70,
    reversed_=-78.8, reversed_t=-3.51,
    static_usfr=16.7, static_usfr_t=0.71,
    static_shv=6.3, static_shv_t=1.40,
    n_switches=333, switches_per_yr=26.9, frac_usfr=55.2,
    # B1a — allocation vs timing (gross), and the placebo re-drawn on 5 seeds (net)
    blend_bp=12.1, blend_t=0.91,
    timing_bp=-18.9, timing_t=-0.97, timing_ci=(-61.9, 18.3),
    placebo_seeds=(-99.6, -88.1, -96.8, -121.4, -93.5),
    # B2 — bootstrap
    ci_gross=(-55.0, 34.9), ci_gross_neg=61.1,
    ci_net=(-169.0, -67.9), ci_net_neg=100.0,
    # B3 — lookback grid
    grid=[
        dict(L=5, gross=-2.5, gt=-0.14, net=-198.2, nt=-9.31, rev=-170.8, sw=604),
        dict(L=21, gross=-6.9, gt=-0.29, net=-115.3, nt=-4.39, rev=-78.8, sw=333),
        dict(L=63, gross=12.1, gt=0.53, net=-50.9, nt=-2.08, rev=-51.3, sw=191),
        dict(L=126, gross=-2.6, gt=-0.12, net=-42.0, nt=-1.88, rev=-13.1, sw=117),
    ],
    # B4 — cost sweep
    costs=[(0.0, -6.9, -0.29), (1.0, -61.1, -2.45), (2.0, -115.3, -4.39),
           (5.0, -277.9, -8.40), (10.0, -549.0, -11.17)],
    # B5 — era cut
    era_e_n=1465, era_e_gross=-38.5, era_e_gt=-0.77, era_e_net=-158.2, era_e_nt=-3.02,
    era_e_usfr=17.2, era_e_usfr_t=0.35, era_e_shv=14.0, era_e_shv_t=2.25,
    era_l_n=1609, era_l_gross=21.8, era_l_gt=1.87, era_l_net=-76.5, era_l_nt=-4.74,
    era_l_usfr=14.6, era_l_usfr_t=1.17, era_l_shv=-1.0, era_l_shv_t=-0.15,
    # B6 — SGOV cross-check
    sg_start="2020-06-01", sg_n=1528,
    sg_sw_gross=11.5, sg_sw_gross_t=1.01, sg_sw_net=-89.5, sg_sw_net_t=-5.48,
    sgov=12.3, sgov_t=3.98, sgov_ci=(8.0, 16.5),
    # B7 — split-half (sample midpoint, non-overlapping) + the one-off swap cost
    sgov_e1=17.0, sgov_e1_t=3.18, sgov_e2=7.7, sgov_e2_t=1.96,
    sgov_e1_ci=(10.2, 23.5), sgov_e2_ci=(3.0, 12.3), sgov_payback_months=3.9,
    sg_usfr=15.7, sg_usfr_t=1.32, sg_shv=-6.8, sg_shv_t=-1.21,
    # synthetic control
    syn_pl_spread=0.222, syn_pl_bp=137.8, syn_pl_t=16.02,
    syn_rw_spread=0.210, syn_rw_bp=9.6, syn_rw_t=5.30,
    syn_nl_spread=-0.0003, syn_nl_sd=0.0015, syn_nl_tmean=-0.77, syn_nl_fire=0,
    real_spread=0.178,
)

HEADER = f"""# Study 923 — The Cash Lag 💤

**Cash vehicles reprice at different speeds. Can you rotate between them for a yield pickup?**

A bill fund holds a ladder, so its yield is roughly the average of the rates at which its
holdings were bought — it inherits a rate change only as the ladder rolls. A floating-rate
note fund, whose coupon resets weekly, inherits it almost at once. The folk conclusion:
**rates rising → sit in the fast repricer (USFR); rates falling → sit in the slow one
(SHV), which keeps a stale-high yield and books a duration gain on top.**

We test both halves on **BIL, SGOV, USFR, SHV** against **^IRX** (the 13-week bill quote),
{R['start']} → {R['end']} ({R['n_days']:,} days), every arm **excess of BIL's own total
return**, 2 bps one-way.

*Real-tape numbers below are the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`); the live cells run the fast synthetic control and are labelled as such.
As-of 2026-06-30.*
"""

SYN_CELL = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "import numpy as np\n"
    "from cash_lag import data, strategy as st\n"
    "\n"
    "# SYNTHETIC ONLY - three worlds, one detector. No real-tape data is touched here.\n"
    "worlds = [\n"
    "    ('real ladder + trending rate ', dict(signal_strength=1.0)),\n"
    "    ('real ladder + random walk   ', dict(signal_strength=1.0, trend_phi=0.0)),\n"
    "    ('no ladder (the null)        ', dict(signal_strength=0.0)),\n"
    "]\n"
    "for label, kw in worlds:\n"
    "    d = st.synthetic_detect(*data.synthetic_panel(seed=923, **kw))\n"
    "    print('%s duration spread %+.3f yr | switch gross %+7.1f bp/yr (t=%+6.2f)'\n"
    "          % (label, d['duration_spread'], d['gross_bp'], d['gross_t']))"
)


def _dur_rows():
    return "\n".join(
        f"| **{v}** | {R['wam'][v]} d | **{R['dur'][v]:+.3f} yr** | {R['dur_t'][v]:+.2f} | {R['dur_n'][v]:,} |"
        for v in ("USFR", "SGOV", "BIL", "SHV")
    )


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Why one cash fund is slower than another\n\n"
           "Imagine two people buying three-month Treasury bills. One buys a fresh bill "
           "every week; after a month, most of what she owns was bought at *today's* rate. "
           "The other bought a year's worth at once; she is still earning last year's rate "
           "and will be for months. Same asset, same safety — different **speed**.\n\n"
           "That speed has a name (weighted average maturity) and a shadow (duration): the "
           "slower fund also gets *marked down* when rates jump, because the old, "
           "lower-yielding bills it holds are suddenly worth less. So a rate rise hurts the "
           "slow fund twice — a mark-down today, and weeks of stale yield afterwards.\n\n"
           "> 🔬 **For the quants** — a ladder of *n*-day bills has duration ≈ *n*/2 days, "
           "so the ordering below is a *prediction* of bond arithmetic, not a discovery."),
        md("## 2. The tape agrees, emphatically\n\n"
           "We measured each fund's realised sensitivity to a move in the 13-week bill rate "
           "— no clever modelling, just daily returns against daily rate changes:\n\n"
           "| Fund | What it holds | Realised duration | HAC *t* |\n|---|---|--:|--:|\n"
           f"| **USFR** | floating-rate notes, coupon resets weekly | **{R['dur']['USFR']:+.3f} yr** | {R['dur_t']['USFR']:+.2f} |\n"
           f"| **SGOV** | 0-3 month bills | **{R['dur']['SGOV']:+.3f} yr** | {R['dur_t']['SGOV']:+.2f} |\n"
           f"| **BIL** | 1-3 month bills | **{R['dur']['BIL']:+.3f} yr** | {R['dur_t']['BIL']:+.2f} |\n"
           f"| **SHV** | 0-1 year Treasuries | **{R['dur']['SHV']:+.3f} yr** | {R['dur_t']['SHV']:+.2f} |\n\n"
           "The order is exactly the maturity order, and every non-zero number is about as "
           "certain as anything in finance ever gets. **USFR's is a clean zero** — a bond "
           "whose coupon resets every week really does not care where rates go."),
        code(
            "R = dict(dur=%r, dur_t=%r)\n"
            "for v in ('USFR','SGOV','BIL','SHV'):\n"
            "    print('%%-5s realised duration %%+.3f yr   (HAC t = %%+5.2f)'\n"
            "          %% (v, R['dur'][v], R['dur_t'][v]))"
            % (R["dur"], R["dur_t"])
        ),
        md("## 3. So the lag is real. Now: is it worth anything?\n\n"
           "The rule writes itself. Rates rising? Hold the fast one, and skip the mark-down. "
           "Rates falling? Hold the slow one, keep its stale-high yield and pocket a small "
           "gain as its old bills become valuable. We checked the direction of the bill rate "
           "over the last month, waited a day (no peeking), and rotated.\n\n"
           f"**Before any trading costs at all, it earned {R['sw_gross']:+.1f} basis points a "
           f"year** — that is {abs(R['sw_gross'])/100:.2f} of one percent, in the *wrong* "
           f"direction, with a *t* of {R['sw_gross_t']:+.2f}. Zero, in other words. Charge a "
           f"realistic two basis points a trade and it becomes **{R['sw_net']:+.1f} bp/yr**."),
        code(
            "R = dict(sw_gross=%r, sw_gross_t=%r, sw_net=%r, sw_net_t=%r,\n"
            "         static_usfr=%r, n_switches=%r, switches_per_yr=%r)\n"
            "print('switch rule, before costs : %%+7.1f bp/yr   (t = %%+5.2f)'\n"
            "      %% (R['sw_gross'], R['sw_gross_t']))\n"
            "print('switch rule, after costs  : %%+7.1f bp/yr   (t = %%+5.2f)'\n"
            "      %% (R['sw_net'], R['sw_net_t']))\n"
            "print('best fund, just held      : %%+7.1f bp/yr' %% R['static_usfr'])\n"
            "print('trades needed             : %%d  (%%.0f a year)'\n"
            "      %% (R['n_switches'], R['switches_per_yr']))"
            % (R["sw_gross"], R["sw_gross_t"], R["sw_net"], R["sw_net_t"],
               R["static_usfr"], R["n_switches"], R["switches_per_yr"])
        ),
        md("## 4. Why it can't work — the prize is smaller than the ticket\n\n"
           f"Here is the whole problem in two numbers. The *entire* gap between the best and "
           f"worst cash fund on this tape is about **{R['static_usfr']:.0f} basis points a "
           f"year**. One round trip — sell one fund, buy another — costs about **4 basis "
           f"points**. The rule trades **{R['switches_per_yr']:.0f} times a year**.\n\n"
           "You are spending roughly a percent of friction chasing a sixth of a percent of "
           "prize. And we know the losses are friction rather than bad luck, because two "
           "controls lose in exactly the same way: running the rule **backwards** loses "
           f"{R['reversed_']:+.0f} bp/yr, and a **random** switch that trades just as often "
           f"loses {R['placebo']:+.0f} bp/yr. Direction has nothing to do with it."),
        md("## 5. The one thing that does work — and it isn't a trade\n\n"
           f"Since 2020, simply **owning SGOV instead of BIL** — no timing, no rotation, one "
           f"decision, ever — has beaten BIL by **{R['sgov']:+.1f} bp/yr** with a *t* of "
           f"**{R['sgov_t']:+.2f}**, and it is positive in both halves of that sample (though "
           "the recent half is only half as big). Part of it is a cheaper fee, part is the "
           "shorter ladder. Buying it costs one round trip — about 4 basis points, paid back "
           f"in **under four months** — and then nothing, ever again.\n\n"
           "One caveat we owe you: we picked SGOV **after** looking at the three candidates, "
           "so its *t*-stat is not the reason to believe it. The reason is that the fee gap "
           "driving it was published in advance and did not need to be discovered.\n\n"
           "> 🔬 **For the quants** — bootstrap CI "
           f"[{R['sgov_ci'][0]:+.1f}, {R['sgov_ci'][1]:+.1f}] bp/yr, clear of zero; "
           f"split-half {R['sgov_e1']:+.1f} (*t* = {R['sgov_e1_t']:+.2f}) then "
           f"{R['sgov_e2']:+.1f} (*t* = {R['sgov_e2_t']:+.2f}). This is a fee-and-maturity "
           "identity, not a risk premium — which is precisely why it survives a "
           "hindsight-selection charge that a return anomaly would not."),
        md("## 6. Is the machine broken? (live synthetic check — no real data)\n\n"
           "Before believing a zero, check that the detector can see anything at all. We "
           "build three make-believe worlds and run the *same* code on them. In the middle "
           "world, rate moves are made completely unpredictable — and the rotation *still* "
           "earns a small, reliable amount, purely because the funds' yields differ. That is "
           "the smallest thing this harness needs to notice, and it notices it easily.\n\n"
           "**The real tape came in below even that floor.**"),
        code(SYN_CELL),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The lag is real, large and ordered exactly as bond "
           f"arithmetic predicts (durations {R['dur']['USFR']:+.3f} → {R['dur']['SHV']:+.3f} "
           f"years, *t* up to {R['dur_t']['SHV']:.1f}). The *rotation* is not: "
           f"{R['sw_gross']:+.1f} bp/yr before costs, *t* = {R['sw_gross_t']:+.2f}. Knowing "
           f"how a fund lags tells you nothing about which to hold next, because last "
           f"month's rate move does not forecast next month's.\n"
           f"- **Tradability — Mirage.** Net {R['sw_net']:+.0f} bp/yr, and a random switch "
           f"that trades as often loses the same — the loss is friction, full stop. The only "
           f"bankable finding is a **fund swap, not a trade**: hold SGOV rather than BIL for "
           f"{R['sgov']:+.1f} bp/yr and then leave it alone."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 923 — The Cash Lag — the teardown\n\n"
           "Window-free duration regressions, the distributed-lag pass-through profile, the "
           "proxy-window artefact sweep, the full pre-registered lookback grid, the cost "
           "sweep, the era cut, the block-bootstrap CI, the SGOV cross-check, and a live "
           "three-world synthetic control. Every real-tape number is frozen from "
           f"`docs/results.md` (Fingerprint `{R['fp']}`); the live cells are synthetic and "
           "labelled as such.\n\n"
           f"**Sample.** BIL ∩ USFR ∩ SHV ∩ ^IRX, {R['start']} → {R['end']}, "
           f"n = {R['n_days']:,}, daily total-return closes (`auto_adjust=True`); ^IRX is a "
           "yield *quote*, not a return. One execution lag: the signal is formed on data "
           "through *t* and acted at *t+1*. Long-only, so no borrow. All arms excess of "
           "BIL's own total return."),
        code("R = %r" % (R,)),
        md("## A1. Realised effective duration — window-free\n\n"
           "Daily total return on daily Δ^IRX (percentage points); duration = −100 × slope, "
           "HAC(5). No trailing window, therefore no averaging artefact.\n\n"
           "| Vehicle | Nominal WAM | Realised duration | HAC *t* | n |\n|---|--:|--:|--:|--:|\n"
           + _dur_rows() +
           "\n\nThe ordering is the published-WAM ordering. USFR's realised rate duration is a "
           "statistical zero — correct for a weekly-resetting FRN. Nominal WAMs are a "
           "**PROXY/ASSUMPTION** (issuer pages) used only to state the a-priori ordering; no "
           "calculation touches them."),
        code(
            "for v in ('USFR','SGOV','BIL','SHV'):\n"
            "    print(f\"{v:5s} WAM {R['wam'][v]:4d}d  duration {R['dur'][v]:+.3f} yr  \"\n"
            "          f\"HAC t {R['dur_t'][v]:+6.2f}  n={R['dur_n'][v]:,}\")\n"
            "print(f\"\\nduration spread USFR->SHV: {R['dur']['SHV'] - R['dur']['USFR']:+.3f} yr\")"
        ),
        md("## A2. Distributed-lag pass-through (21-day realised-yield proxy)\n\n"
           "Δ₂₁ of the realised-yield proxy on Δ₂₁^IRX at lags 0, 7, … 63, HAC(42). The proxy "
           "is **labelled**: trailing 21-day total return annualised, in percent — not an SEC "
           "yield. Σβ is eventual pass-through; β at lag 0 is the duration shock."),
        code(
            "print(f\"{'':6s}{'sum(b)':>8s}{'b(lag0)':>10s}{'t':>8s}{'peak':>7s}{'centroid':>10s}\")\n"
            "for v in ('SGOV','BIL','USFR','SHV'):\n"
            "    print(f\"{v:6s}{R['sum_beta'][v]:+8.2f}{R['b0'][v]:+10.2f}\"\n"
            "          f\"{R['b0_t'][v]:+8.2f}{R['peak'][v]:6d}d{R['centroid'][v]:9.1f}d\")\n"
            "print('\\nSum(beta) ~ 1 everywhere: a bill portfolio eventually inherits the whole bill rate.')\n"
            "print('b(lag0) scales with WAM: SHV -3.26 >> BIL -1.16 > SGOV -0.80 >> USFR ~0 (all noise).')"
        ),
        md("## A3. How much of the lag is the ruler?\n\n"
           "A trailing *w*-day return averages the rate over *w* days, so part of the measured "
           "lag is the instrument. If the centroid rose one-for-one with *w*, the whole "
           "\"lag\" would be an artefact."),
        code(
            "print(f\"{'window':>8s}\" + ''.join(f'{v:>8s}' for v in ('SGOV','BIL','USFR','SHV')))\n"
            "for w in (5, 10, 21, 42):\n"
            "    row = R['win_sweep'][w]\n"
            "    print(f\"{w:6d}d \" + ''.join(f\"{row[v]:8.1f}\" for v in ('SGOV','BIL','USFR','SHV')))\n"
            "print('\\nThe centroid drifts up with the window, so \"27-day lag\" overstates it.')\n"
            "print('But at w=5 the bill funds still sit 13-19d behind - 2-4x the ruler.')\n"
            "print('Residual repricing delay: roughly two to four weeks. USFR\\'s column is noise.')"
        ),
        md("## A4. Is the measurement era-robust? (the half that carries the stamp)\n\n"
           "The trade gets an era cut in B5; the measurement has to face the same test. Same "
           "split (2020-01-01), same window-free regression. Duration in years, HAC(5) *t*."),
        code(
            "print(f\"{'era':>10s}\" + ''.join(f'{v:>22s}' for v in ('USFR','BIL','SHV')))\n"
            "for era in ('2014-2019', '2020-2026'):\n"
            "    row = R['dur_era'][era]\n"
            "    print(f\"{era:>10s}\" + ''.join(f\"{row[v][0]:+12.3f} (t{row[v][1]:+5.2f})\"\n"
            "                                   for v in ('USFR','BIL','SHV')))\n"
            "print('\\nThe ordering USFR < BIL < SHV holds in BOTH halves, both bill funds clear')\n"
            "print('|t|=2.8 in both, and USFR is a statistical zero in both. The measurement does')\n"
            "print('not depend on the rate cycle - which is what arithmetic, not forecasting, does.')"
        ),
        md("## B1. The rotation and its controls (21d lookback, 2 bps one-way)\n\n"
           "All excess of BIL. `exSharpe` on a cash-vs-cash difference has a denominator of "
           "tens of bp — reported for completeness, never leaned on. The honest units are "
           "annualised bp and the HAC *t*."),
        code(
            "rows = [('switch  GROSS', R['sw_gross'], R['sw_gross_t']),\n"
            "        ('switch  net  ', R['sw_net'], R['sw_net_t']),\n"
            "        ('placebo (turnover-matched random)', R['placebo'], R['placebo_t']),\n"
            "        ('reversed rule', R['reversed_'], R['reversed_t']),\n"
            "        ('static USFR  ', R['static_usfr'], R['static_usfr_t']),\n"
            "        ('static SHV   ', R['static_shv'], R['static_shv_t'])]\n"
            "for tag, bp, t in rows:\n"
            "    print(f\"{tag:34s}{bp:+9.1f} bp/yr   HAC t {t:+6.2f}\")\n"
            "print(f\"\\n{R['n_switches']} switches over 12.4y ({R['switches_per_yr']:.1f}/yr), \"\n"
            "      f\"{R['frac_usfr']:.1f}% of days in USFR\")\n"
            "print('The reversed rule and a turnover-matched random switch lose the SAME way:')\n"
            "print('the loss is friction, not a wrong-way directional bet.')"
        ),
        md("## B1a. Allocation or timing?\n\n"
           "The rule stands in USFR 55.2% of the time, and the two legs do not earn the same "
           "thing — so part of any *gross* number is simply where it happened to stand. "
           "Splitting it: a passive 55/45 USFR/SHV blend that never trades, versus the timing "
           "decision alone (rule − that blend). Neither piece clears |*t*| = 1, and the only "
           "positive component is the one that required no forecast."),
        code(
            "print(f\"passive 55/45 blend, no trading : {R['blend_bp']:+7.1f} bp/yr  \"\n"
            "      f\"HAC t {R['blend_t']:+5.2f}\")\n"
            "print(f\"timing decision alone           : {R['timing_bp']:+7.1f} bp/yr  \"\n"
            "      f\"HAC t {R['timing_t']:+5.2f}   95% CI [{R['timing_ci'][0]:+.1f}, \"\n"
            "      f\"{R['timing_ci'][1]:+.1f}]\")\n"
            "print(f\"= switch rule, gross            : {R['sw_gross']:+7.1f} bp/yr  \"\n"
            "      f\"HAC t {R['sw_gross_t']:+5.2f}\")\n"
            "print()\n"
            "print('placebo re-drawn on 5 seeds (net): '\n"
            "      + '  '.join('%+.1f' % b for b in R['placebo_seeds']))\n"
            "print('range [%+.1f, %+.1f] bp/yr - it brackets the real rule at %+.1f.'\n"
            "      % (min(R['placebo_seeds']), max(R['placebo_seeds']), R['sw_net']))"
        ),
        md("## B2. Block-bootstrap CI on the annualised excess (2,000 draws, 21-day blocks)"),
        code(
            "print(f\"gross    : {R['sw_gross']:+7.1f} bp/yr  95% CI [{R['ci_gross'][0]:+.1f}, \"\n"
            "      f\"{R['ci_gross'][1]:+.1f}]  share<0 {R['ci_gross_neg']:.1f}%\")\n"
            "print(f\"net 2bp  : {R['sw_net']:+7.1f} bp/yr  95% CI [{R['ci_net'][0]:+.1f}, \"\n"
            "      f\"{R['ci_net'][1]:+.1f}]  share<0 {R['ci_net_neg']:.1f}%\")"
        ),
        md("## B3. The whole pre-registered lookback grid\n\n"
           "Four windows specified before running, all four reported. With four looks a "
           "nominal |*t*| ≈ 2 on the best would mean little (Harvey-Liu-Zhu 2016) — and "
           "there is no best."),
        code(
            "print(f\"{'L':>5s}{'gross':>10s}{'t':>8s}{'net':>10s}{'t':>8s}{'reversed':>11s}{'sw':>7s}\")\n"
            "for g in R['grid']:\n"
            "    print(f\"{g['L']:4d}d{g['gross']:+10.1f}{g['gt']:+8.2f}\"\n"
            "          f\"{g['net']:+10.1f}{g['nt']:+8.2f}{g['rev']:+11.1f}{g['sw']:7d}\")\n"
            "print(f\"\\nlargest gross |t| anywhere in the grid: \"\n"
            "      f\"{max(abs(g['gt']) for g in R['grid']):.2f}\")"
        ),
        md("## B4. Cost sweep — the 2 bps one-way figure is a PROXY/ASSUMPTION\n\n"
           "Swept 0-10 bps. It cannot rescue the rule because the *gross* edge is already "
           "zero: at literally free trading the rule earns nothing."),
        code(
            "for c, bp, t in R['costs']:\n"
            "    tag = '  <- gross' if c == 0 else ''\n"
            "    print(f\"cost {c:5.1f} bps  {bp:+9.1f} bp/yr   HAC t {t:+7.2f}{tag}\")"
        ),
        md("## B5. Era cut (split 2020-01-01)\n\n"
           "2014-2019 spans one gentle hiking cycle; 2020-2026 the zero floor, the 0→5.4% "
           "spike and the cuts."),
        code(
            "print(f\"2014-2019 (n={R['era_e_n']:,}): gross {R['era_e_gross']:+7.1f} (t={R['era_e_gt']:+5.2f})  \"\n"
            "      f\"net {R['era_e_net']:+7.1f} (t={R['era_e_nt']:+5.2f})  |  \"\n"
            "      f\"USFR {R['era_e_usfr']:+5.1f} (t={R['era_e_usfr_t']:+5.2f})  \"\n"
            "      f\"SHV {R['era_e_shv']:+5.1f} (t={R['era_e_shv_t']:+5.2f})\")\n"
            "print(f\"2020-2026 (n={R['era_l_n']:,}): gross {R['era_l_gross']:+7.1f} (t={R['era_l_gt']:+5.2f})  \"\n"
            "      f\"net {R['era_l_net']:+7.1f} (t={R['era_l_nt']:+5.2f})  |  \"\n"
            "      f\"USFR {R['era_l_usfr']:+5.1f} (t={R['era_l_usfr_t']:+5.2f})  \"\n"
            "      f\"SHV {R['era_l_shv']:+5.1f} (t={R['era_l_shv_t']:+5.2f})\")\n"
            "print('\\nThe gross edge FLIPS SIGN between eras and clears |t|=2 in neither.')\n"
            "print('SHV\\'s own static premium is era-bound: +14.0 (t=+2.25) then -1.0 - the')\n"
            "print('2022-23 hiking cycle punished its extra duration.')"
        ),
        md(f"## B6. SGOV four-vehicle cross-check ({R['sg_start']} → {R['end']}, n = {R['sg_n']:,})\n\n"
           "**Survivorship note.** These are the four *surviving* large cash ETFs; SGOV and "
           "USFR are young and were launched into a world already selecting for "
           "cheap-and-short. The dispersion measured here is an upper bound on what a 2014 "
           "investor could have chosen between.\n\n"
           "**Selection, named.** SGOV is the *best of the three static arms below* and it was "
           "chosen **after** they were printed. Its *t* = +3.98 is therefore one look out of "
           "three and buys nothing on its own; what defends it is that the mechanism — a "
           "published, ex-ante-knowable expense-ratio gap plus a shorter ladder — did not have "
           "to be discovered on this tape, and that it survives a split-half cut. It is also "
           "**not the claim this study tested**."),
        code(
            "print(f\"switch gross {R['sg_sw_gross']:+6.1f} bp/yr (t={R['sg_sw_gross_t']:+5.2f})   \"\n"
            "      f\"net {R['sg_sw_net']:+6.1f} (t={R['sg_sw_net_t']:+5.2f})\")\n"
            "print(f\"static SGOV  {R['sgov']:+6.1f} bp/yr (t={R['sgov_t']:+5.2f})  \"\n"
            "      f\"boot CI [{R['sgov_ci'][0]:+.1f}, {R['sgov_ci'][1]:+.1f}]   <- the only robust number here\")\n"
            "print(f\"static USFR  {R['sg_usfr']:+6.1f} bp/yr (t={R['sg_usfr_t']:+5.2f})\")\n"
            "print(f\"static SHV   {R['sg_shv']:+6.1f} bp/yr (t={R['sg_shv_t']:+5.2f})\")\n"
            "print(f\"\\nSGOV split-half: {R['sgov_e1']:+.1f} (t={R['sgov_e1_t']:+.2f}, CI [{R['sgov_e1_ci'][0]:+.1f}, {R['sgov_e1_ci'][1]:+.1f}])  then \"\n"
            "      f\"{R['sgov_e2']:+.1f} (t={R['sgov_e2_t']:+.2f}, CI [{R['sgov_e2_ci'][0]:+.1f}, {R['sgov_e2_ci'][1]:+.1f}])\")\n"
            "print('both halves positive, but the effect HALVES - the late half is only t=+%.2f.' % R['sgov_e2_t'])\n"
            "print('one-off swap: 2 legs x 2bp = 4bp, paid back in %.1f months, vs 27 round trips a year for the rotation.'\n"
            "      % R['sgov_payback_months'])\n"
            "print('SELECTED EX POST from the three arms above - the ex-ante fee gap is the defence, not the t.')\n"
            "print('A fee-and-WAM identity (published ERs put BIL ~4-5bp above SGOV, an ASSUMPTION),')\n"
            "print('not a risk premium - which is exactly why it survives. Zero timing required.')"
        ),
        md("## Synthetic control — the power floor (live, synthetic only)\n\n"
           "The panel plants a WAM ladder and a trending rate path **independently**, so each "
           "can be switched off. The middle world is the one that matters: with rate changes "
           "made unpredictable, the rotation *still* earns a small, reliably positive gross "
           "excess — pure carry rotation, no forecasting. That is the smallest effect this "
           "harness must be able to see."),
        code(SYN_CELL),
        code(
            "nl = np.array([st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=923+s))['gross_t']\n"
            "               for s in range(8)])\n"
            "print('null x8 gross t: mean %+.2f (sd %.2f), |t|>=2 on %d/8 seeds'\n"
            "      % (nl.mean(), nl.std(ddof=1), int((abs(nl) >= 2).sum())))\n"
            "print()\n"
            "print('planted   : spread %+.3f yr, gross %+7.1f bp/yr (t=%+6.2f)'\n"
            "      % (R['syn_pl_spread'], R['syn_pl_bp'], R['syn_pl_t']))\n"
            "print('power floor: spread %+.3f yr, gross %+7.1f bp/yr (t=%+6.2f)  <- no forecasting at all'\n"
            "      % (R['syn_rw_spread'], R['syn_rw_bp'], R['syn_rw_t']))\n"
            "print('REAL TAPE  : spread %+.3f yr, gross %+7.1f bp/yr (t=%+6.2f)  <- below the floor'\n"
            "      % (R['real_spread'], R['sw_gross'], R['sw_gross_t']))"
        ),
        md("> 💡 **In plain words** — the real funds' duration spread (0.178 yr) is just as "
           "wide as the one we planted (0.21 yr), and the harness finds a 9.6 bp/yr edge in "
           "the planted world at *t* = +5.3 without any forecasting ability whatsoever. It "
           "finds −6.9 bp/yr at *t* = −0.29 on the real tape. The machine has power to spare; "
           "the tape has nothing to give it."),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** *Measurement:* real and overwhelming. Realised durations "
           f"{R['dur']['USFR']:+.3f} (USFR), {R['dur']['SGOV']:+.3f} (SGOV), "
           f"{R['dur']['BIL']:+.3f} (BIL), {R['dur']['SHV']:+.3f} yr (SHV), |*t*| up to "
           f"{R['dur_t']['SHV']:.1f}, ordered exactly by WAM; Σβ ≈ 1 (full pass-through); "
           f"a residual repricing delay of two to four weeks survives the window-artefact "
           f"sweep, and the ordering holds at |*t*| ≥ 2.8 in **both** eras (A4) — but it is "
           f"`duration ≈ WAM/2`, arithmetic measured well, not a forecast. "
           f"*Rotation:* absent. {R['sw_gross']:+.1f} bp/yr gross "
           f"(HAC *t* = {R['sw_gross_t']:+.2f}), bootstrap CI "
           f"[{R['ci_gross'][0]:+.0f}, {R['ci_gross'][1]:+.0f}] straddling zero, max |*t*| = "
           f"{max(abs(g['gt']) for g in R['grid']):.2f} across the pre-registered grid, and "
           f"the gross sign flips between eras. The lag is a fact about the instruments; it "
           f"is not information about the future.\n"
           f"- **Tradability — Mirage.** The prize is bounded by ~16 bp/yr of cross-vehicle "
           f"dispersion; a round trip costs 4 bps and the rule trades "
           f"{R['switches_per_yr']:.0f}×/yr. Net {R['sw_net']:+.0f} bp/yr at 2 bps, still "
           f"{R['costs'][1][1]:+.0f} at 1 bp, and the turnover-matched placebo "
           f"({R['placebo']:+.0f}) and reversed rule ({R['reversed_']:+.0f}) lose the same — "
           f"identifying the loss as friction. The only bankable result is a **fund swap, not "
           f"a trade**: SGOV over BIL, {R['sgov']:+.1f} bp/yr (*t* = {R['sgov_t']:+.2f}, CI "
           f"clear of zero, both halves positive but the late one only {R['sgov_e2']:+.1f} at "
           f"*t* = {R['sgov_e2_t']:+.2f}), a fee-and-WAM identity requiring no timing and one "
           f"4 bp round trip. Named: that arm was **selected ex post** from three static "
           f"candidates, and it is not the claim this study tested."),
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
