"""Generate the two narrative notebooks for Study 957 (Holdco Discount).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the numbers are baked into the
generated cells as literals, so a reader can see exactly what was published. The only live
cells run the fast synthetic control, and they are never presented under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. Seven listed holdcos vs their
# dominant listed stakes, 2004-01-02 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2004-01-02", end="2026-06-30", n_names=7, fp="d51337b2a3bb",
    # (a) mean reversion
    pooled_beta=-0.0002, pooled_t=-0.09, pooled_n=24572, pooled_days=5581, pooled_r2=0.0000,
    n_neg=5,
    prosus_beta=-0.0358, prosus_t=-4.07, prosus_n=1219,
    dior_beta=0.0037, lbrd_beta=-0.0031, bol_beta=0.0031,
    bol_hl=1514, heio_hl=95,
    # (b) the pair race
    timed_sharpe=0.312, timed_cagr=2.01, timed_vol=7.22, timed_dd=-17.9, timed_t=1.69,
    timed_gross=0.566, timed_gross_t=3.04, timed_inv=80.7,
    always_sharpe=-0.268, always_cagr=-1.68, always_dd=-43.8, always_t=-1.41,
    always_gross=0.046, always_gross_t=0.24,
    adv=0.580, t_diff=3.25,
    ci_net_lo=-0.050, ci_net_hi=0.656, ci_net_neg=4.7,
    ci_gross_lo=0.204, ci_gross_hi=0.910,
    ci_always_lo=-0.645, ci_always_hi=0.093, ci_always_neg=93.0,
    # eras
    era_e_sh=0.319, era_e_t=1.33, era_l_sh=0.256, era_l_t=0.86,
    # sweeps
    cost0=0.510, cost0_t=2.75, cost5=0.411, cost5_t=2.22,
    cost25=0.016, cost25_t=0.08, cost50=-0.467, cost50_t=-2.52,
    borrow0=0.367, borrow0_t=1.99, borrow600=0.036, borrow600_t=0.20,
    # the deciding check
    clean_net=0.148, clean_net_t=0.76, clean_gross=0.368, clean_gross_t=1.87,
    clean_mr_beta=0.0033, clean_mr_t=1.27,
    adr_net=0.389, adr_net_t=1.85, adr_gross=0.458, adr_gross_t=2.16,
    # per-name standalone
    pn_prosus=0.661, pn_prosus_t=1.61, pn_softbank=0.398, pn_softbank_t=1.17,
    pn_heio=0.145, pn_dior=0.087, pn_lbrd=0.066, pn_bol=0.093,
    # robustness
    loo_lo=0.259, loo_hi=0.346, loo_t_lo=1.41, loo_t_hi=1.87,
    calib_lo=0.275, calib_hi=0.321,
    lt_net=-0.040, lt_net_t=-0.24, lt_gross=0.174,
    lo_timed=0.536, lo_holdcos=0.655, lo_stakes=0.636, lo_t_diff=0.02,
    # free-parameter sweeps (added by the audit)
    thr_net_lo=0.08, thr_net_hi=1.90, thr_gross_lo=1.56, thr_gross_hi=3.43,
    thr_cells=17, thr_cells_net2=0, thr_best_gross=0.653, thr_best_gross_t=3.43,
    zw252_t=2.42, zw378_t=2.46, zw756_t=1.89, zw1008_t=0.94,
    zw756=0.333, zw1008=0.160,
    lbrd_keep_beta=0.0005, lbrd_keep_t=0.19, lbrd_keep_neg=4, lbrd_cut_days=406,
    irx_cagr=1.45, bil_cagr=1.36,
    # synthetic control
    syn_pl_t=-11.76, syn_pl_gross=0.999, syn_nl_t=-0.38, syn_nl_gross=0.129,
)


HEADER = f"""# Study 957 — Holdco Discount 🏛️

**Buy a conglomerate below the sum of its parts — does the gap ever close?**

Some listed holding companies are almost pure wrappers around **one listed stake**. Christian
Dior is mostly LVMH. Heineken Holding is Heineken NV and nothing else. Liberty Broadband was
Charter. That makes them the rare case where you can mark net asset value **from the tape**:
multiply the stake's price by the number of shares the holdco owns, and compare it with what
the holdco itself costs. The gap is the **holdco discount**, and it is almost always there.

The pitch writes itself: you are buying EUR 1 of LVMH for 79 cents. The question this study
asks is the only one that matters — *does the missing 21 cents ever come back?*

We build the discount for **{R['n_names']} holdcos** from price-only closes,
{R['start']} → {R['end']}, and run two tests: does the gap mean-revert, and does buying it
wide — hedged, costed, with borrow charged on the short — pay.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the live
cells run the offline synthetic control only. As-of 2026-06-30.*
"""

SYN_SETUP = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "from holdco_nav import data, strategy as st\n"
)


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),

        md("## 1. What the discount looks like\n\n"
           "Seven holdcos, seven long-running gaps. Note how different they are — Heineken "
           "Holding sits quietly around 12%, Bollore drifted out past 70%. That range is the "
           "first clue: if these were *mistakes*, they would look alike, and they would close."),
        code(
            "panel = [\n"
            "    ('Heineken Holding',  'Heineken NV', 11.7,  4.2,   95),\n"
            "    ('Christian Dior',    'LVMH',        14.7,  9.1,  232),\n"
            "    ('Liberty Broadband', 'Charter',      6.2, 19.4,  184),\n"
            "    ('Naspers',           'Tencent',     37.9, 19.5,  329),\n"
            "    ('Prosus',            'Tencent',     36.3,  7.7,   94),\n"
            "    ('Bollore',           'Vivendi',     74.4, 13.7, 1514),\n"
            "    ('SoftBank',          'Alibaba',     59.9,  8.7,  105),\n"
            "]\n"
            "print('holdco             its main stake   avg gap   swing   half-life')\n"
            "for h, s, mean, sd, hl in panel:\n"
            "    print('%-18s %-14s %6.1f%% %6.1f%% %8d d' % (h, s, mean, sd, hl))"
        ),
        md("> 🔬 **For the quants** — \"half-life\" is the AR(1) half-life of the raw discount. "
           "Bollore's is **1,514 trading days**: six years. A gap that takes six years to "
           "half-close is not a trade, it is a career."),

        md("## 2. Test one — does a wide gap narrow?\n\n"
           "The clean way to ask it: when a holdco's discount is unusually wide *for that "
           "holdco* — a standard deviation above its own two-year average — what does the gap "
           "look like six months later? If the folklore is right, it should be narrower."),
        code(
            "print('pooled slope on " + f"{R['pooled_n']:,}" + " observations : "
            + f"{R['pooled_beta']:+.4f}" + "')\n"
            "print('  (negative would mean the gap closes)')\n"
            "print('HAC t-statistic                     : " + f"{R['pooled_t']:+.2f}" + "')\n"
            "print('R-squared                           : " + f"{R['pooled_r2']:.4f}" + "')\n"
            "print('names with the RIGHT sign           : " + f"{R['n_neg']}" + " of 7')"
        ),
        md(f"**Nothing.** The slope is statistically invisible "
           f"(*t* = {R['pooled_t']:+.2f}, R² = {R['pooled_r2']:.4f}). Knowing that a holdco is "
           f"unusually cheap against its own history tells you essentially nothing about where "
           f"the gap will be in six months. **{R['n_neg']} of 7** names lean the right way and "
           f"two lean the wrong way — Christian Dior and Bollore both had wide gaps get "
           f"*wider* — which is exactly what a coin flip looks like."),

        md("## 3. Test two — but does trading it pay anyway?\n\n"
           "A gap does not have to be predictable *on average* to be tradable *at the extremes*. "
           "So: buy the holdco and short its stake one-for-one — that isolates the gap and "
           "nothing else — but only when the gap is unusually wide, and let go once it is back to "
           "normal. Charge 10 bps a leg, and 1% a year to borrow the short."),
        code(
            "print('gross of costs                    : Sharpe "
            + f"{R['timed_gross']:+.3f}  (t = {R['timed_gross_t']:+.2f})" + "   <- looks like something')\n"
            "print('after costs and borrow            : Sharpe "
            + f"{R['timed_sharpe']:+.3f}  (t = {R['timed_t']:+.2f})" + "')\n"
            "print('just owning the gap, never timing : Sharpe "
            + f"{R['always_sharpe']:+.3f}  (t = {R['always_t']:+.2f})" + "')"
        ),
        md(f"Two things jump out. First, **just owning the discount and waiting lost money** — "
           f"Sharpe {R['always_sharpe']:+.3f}, CAGR {R['always_cagr']:+.2f}% a year across two "
           f"decades. The gaps did not close; on balance they widened. Second, the *timed* "
           f"version does look alive before costs. Which is where the real work starts."),

        md("## 4. The check that decides it\n\n"
           "Three of our seven pairs trade as **thin over-the-counter ADRs** in New York — "
           "Naspers, Prosus and SoftBank. Their closing prints can be hours stale, or simply not "
           "refreshed at all. And a stale price on one leg of a long/short pair *invents* exactly "
           "the pattern we are hunting: an apparent gap today that \"closes\" tomorrow when the "
           "quote finally catches up.\n\n"
           "So split the panel. Four pairs where both legs print in the same liquid session; "
           "three where one leg is a stale OTC quote."),
        code(
            "rows = [\n"
            "    ('full panel (all 7)', "
            + f"{R['timed_gross']:.3f}, {R['timed_gross_t']:.2f}, {R['timed_sharpe']:.3f}, {R['timed_t']:.2f}" + "),\n"
            "    ('primary listings only (4)', "
            + f"{R['clean_gross']:.3f}, {R['clean_gross_t']:.2f}, {R['clean_net']:.3f}, {R['clean_net_t']:.2f}" + "),\n"
            "    ('the 3 thin OTC ADR pairs', "
            + f"{R['adr_gross']:.3f}, {R['adr_gross_t']:.2f}, {R['adr_net']:.3f}, {R['adr_net_t']:.2f}" + "),\n"
            "]\n"
            "print('%-28s %9s %7s %9s %7s' % ('sub-panel', 'gross', '(t)', 'net', '(t)'))\n"
            "for tag, gs, gt, ns, nt in rows:\n"
            "    print('%-28s %+9.3f %+7.2f %+9.3f %+7.2f' % (tag, gs, gt, ns, nt))"
        ),
        md(f"There it is. Take away the stale-quote names and the gross *t* collapses from "
           f"**{R['timed_gross_t']:+.2f} to {R['clean_gross_t']:+.2f}**, and after costs to "
           f"**{R['clean_net_t']:+.2f}**. What looked like a discount edge was largely the least "
           f"liquid corner of the panel bouncing off its own stale prints."),

        md("## 5. And it was fragile anyway\n\n"
           "Even taking the full-panel number at face value, it does not survive contact with a "
           "trading desk."),
        code(
            "rows = [(0, "
            + f"{R['cost0']:.3f}, {R['cost0_t']:.2f}" + "), (5, "
            + f"{R['cost5']:.3f}, {R['cost5_t']:.2f}" + "), (10, "
            + f"{R['timed_sharpe']:.3f}, {R['timed_t']:.2f}" + "), (25, "
            + f"{R['cost25']:.3f}, {R['cost25_t']:.2f}" + "), (50, "
            + f"{R['cost50']:.3f}, {R['cost50_t']:.2f}" + ")]\n"
            "print('cost per leg   ->  Sharpe (t)')\n"
            "for c, sh, t in rows:\n"
            "    flag = '   <- our assumption' if c == 10 else ''\n"
            "    print('  %2d bps          %+.3f (%+.2f)%s' % (c, sh, t, flag))"
        ),
        md(f"Flat at 25 bps a leg, {R['cost50']:+.2f} Sharpe at 50. This is a two-legged trade in "
           f"OTC ADRs and a Paris small-cap; 10 bps is the *optimistic* end of the range. And no "
           f"individual holdco clears the bar on its own — the best are Prosus "
           f"({R['pn_prosus']:+.2f}, *t* = {R['pn_prosus_t']:+.2f}) and SoftBank "
           f"({R['pn_softbank']:+.2f}, *t* = {R['pn_softbank_t']:+.2f}), which are, again, the two "
           f"thinnest ADRs."),

        md("Then there is the honest question any backtest has to answer: *how much of this "
           "depends on the two dials we chose?* The rule says buy when the gap is one standard "
           "deviation wide and sell when it is back to normal. Those two numbers are arbitrary. "
           "So try every sensible pair of them."),
        code(
            "print('how wide before we buy, how normal before we sell:')\n"
            "print('  %d combinations tried' % "
            + f"{R['thr_cells']}" + ")\n"
            "print('  before costs, the t-stat runs from "
            + f"{R['thr_gross_lo']:+.2f} to {R['thr_gross_hi']:+.2f}" + "')\n"
            "print('  AFTER costs, it runs from "
            + f"{R['thr_net_lo']:+.2f} to {R['thr_net_hi']:+.2f}" + "')\n"
            "print('  combinations that clear the bar (t >= 2) after costs: "
            + f"{R['thr_cells_net2']} of {R['thr_cells']}" + "')"
        ),
        md(f"That is the cleanest statement in the study. Our reported setting is not the best "
           f"one in the grid — a slightly lazier exit does better ({R['thr_best_gross']:+.3f}, "
           f"*t* = {R['thr_best_gross_t']:+.2f}) — so nothing was cherry-picked. But **there is "
           f"no setting of the dials at which this trade is significant after costs**. Not one "
           f"of {R['thr_cells']}."),

        md("## 6. Is the harness even capable of finding this? (live, offline)\n\n"
           "Fair question. So we build a synthetic world where the discount genuinely *does* "
           "mean-revert, and a second where it is a coin-flip random walk, and run the identical "
           "code on both. If the detector fires on the first and stays quiet on the second, then "
           "the flat answer above is a fact about holdcos, not a broken test."),
        code(
            SYN_SETUP +
            "planted = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=957)[0])\n"
            "null    = st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=957)[0])\n"
            "print('a world where the gap DOES close : slope t = %+.2f, gross Sharpe %+.2f'\n"
            "      % (planted['pooled_t'], planted['sharpe_timed_gross']))\n"
            "print('a world where it is a coin flip  : slope t = %+.2f, gross Sharpe %+.2f'\n"
            "      % (null['pooled_t'], null['sharpe_timed_gross']))"
        ),
        md("The machinery works. It finds reversion when reversion is planted, and finds nothing "
           "when there is nothing. The real tape simply has nothing to find."),

        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The discount does **not** mean-revert: a gap one standard "
           f"deviation wider than its own two-year norm predicts nothing six months out "
           f"(*t* = {R['pooled_t']:+.2f}, right sign in {R['n_neg']}/7 names — a coin flip). The "
           f"trading version looks alive gross (*t* = {R['timed_gross_t']:+.2f}) but loses that "
           f"once you charge realistic friction (*t* = {R['timed_t']:+.2f}) and again once you "
           f"remove the three thin OTC ADRs whose stale closes manufacture the pattern "
           f"({R['clean_gross_t']:+.2f} gross, {R['clean_net_t']:+.2f} net) — and it is "
           f"significant after costs at **none** of the {R['thr_cells']} entry/exit settings we "
           f"tried.\n"
           f"- **Tradability — Mirage.** Flat at 25 bps a leg, negative at 50, a bootstrap "
           f"confidence interval that includes zero, and a short leg you would have to borrow in "
           f"exactly the illiquid names producing the number. Meanwhile the patient version — buy "
           f"the discount and wait — lost {abs(R['always_cagr']):.2f}% a year for 22 years.\n"
           f"- **What the gap really is.** Dividend tax leaking up the chain, a family voting "
           f"block that makes a takeover impossible, holding-company overhead, and the risk that "
           f"management reinvests your money badly. Those are reasons a thing is *worth* less, "
           f"not reasons it is *priced wrong* — and they have no reason to go away on your "
           f"schedule. Bollore's discount took six years to half-close, and it went the wrong "
           f"way."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 957 — Holdco Discount — the teardown\n\n"
           "Seven listed holding companies whose value is dominated by one **listed** stake, so "
           "NAV is markable from the tape. Two questions in order: (a) does the observable-NAV "
           "discount mean-revert, (b) does a hedged buy-the-wide-discount rule pay after costs "
           "and borrow. Inside: the NAV construction and its assumptions, the Driscoll-Kraay "
           "pooled predictive regression, the dollar-neutral pair race against an always-on "
           "control, bootstrap CIs, an era cut, cost and borrow sweeps, a NAV-calibration sweep, "
           "leave-one-out, the stale-quote decomposition that decides the verdict, and the live "
           "synthetic control.\n\n"
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`)." % R["fp"]),
        code("R = " + repr(R)),

        md("## Construction, and what is an assumption\n\n"
           "`discount_t = 1 - P_holdco_t / (k * P_stake_t + o)` from **price-only** "
           "(split-adjusted, dividend-unadjusted) closes — putting total-return closes on both "
           "legs of a *valuation* ratio would inject the two names' dividend-yield difference "
           "straight into the \"discount\". Every **return** in the backtest comes from "
           "total-return closes.\n\n"
           "`k` and `o` are **assumptions**, frozen at a 2026 snapshot, and in reality both "
           "drift. Three are published share counts (Heineken Holding, Christian Dior, Liberty "
           "Broadband); four are *anchored* to a widely reported discount on one date, because "
           "ADR ratios and chained holdings make a bottom-up count unverifiable from the tape. "
           "The consequence is stated rather than hidden: the **level** of each series is only as "
           "good as that table, so every test runs on the **trailing z** — each name standardised "
           "against its own preceding 504 days, using only data through *t*. A constant error in "
           "`k` or `o` barely moves a z-score, and the calibration sweep below shows it does not. "
           "For the four names whose `o` is zero the cancellation is *exact* — scale `k` and the "
           "z-score is literally unchanged — so anchoring `k` to a reported discount on one date "
           "cannot leak hindsight into any test.\n\n"
           "Three names stop early, all on **corporate events announced in advance**, never on "
           "performance, and all three shorten the sample: Bollore at the December 2024 Vivendi "
           "four-way split, SoftBank at the 2022 Alibaba disposal, and Liberty Broadband at the "
           "2024-11-13 Charter merger agreement — after which its gap is a **merger spread** "
           "with a contractual exchange ratio and a convergence date, which is Study 366's "
           "subject and not a holdco discount. That last cut removes the one window in the panel "
           "where convergence was *guaranteed*, so it is the conservative choice; keeping it "
           "leaves the pair identical to three decimals and moves the pooled slope to "
           f"{R['lbrd_keep_beta']:+.4f} (*t* = {R['lbrd_keep_t']:+.2f}).\n\n"
           "> 💡 **In plain words** — we cannot pin down exactly how big each discount is, so we "
           "never rely on that. We only ever ask whether *this* holdco is unusually cheap "
           "compared with *its own* recent history."),

        md("## (a) The predictive regression\n\n"
           "`d_{t+126} - d_t = a + b * z_t`, pooled across names. Two problems would wreck a "
           "naive *t*: the forward windows overlap (residuals autocorrelated out to 126 lags), "
           "and the names are contemporaneously correlated — two of them literally sit on the "
           "same underlying, Tencent. So the score contributions are summed **across names within "
           "each calendar day** and only then Bartlett-weighted: Driscoll-Kraay, not a "
           "concatenated Newey-West that would splice one name's last day onto another's first."),
        code(
            "print('pooled beta %+.4f   DK t %+.2f   R2 %.4f   n=%d over %d days'\n"
            "      % (R['pooled_beta'], R['pooled_t'], R['pooled_r2'], R['pooled_n'], R['pooled_days']))\n"
            "print('right sign (beta < 0) in %d/7 names' % R['n_neg'])\n"
            "print()\n"
            "print('the only individually significant name:')\n"
            "print('  Prosus  beta %+.4f  t %+.2f  n=%d  <- the SHORTEST sample in the panel'\n"
            "      % (R['prosus_beta'], R['prosus_t'], R['prosus_n']))\n"
            "print('the two with the WRONG sign:')\n"
            "print('  Christian Dior %+.4f   Bollore %+.4f'\n"
            "      % (R['dior_beta'], R['bol_beta']))\n"
            "print()\n"
            "print('horizon is a convention, not a result -- pooled DK t by forward window:')\n"
            "print('  h= 21d  t -1.18    h= 63d  t -0.40    h=126d  t %+.2f    h=252d  t +0.30'\n"
            "      % R['pooled_t'])"
        ),
        md(f"Pooled *t* = **{R['pooled_t']:+.2f}** and R² = "
           f"{R['pooled_r2']:.4f}. Prosus's *t* = {R['prosus_t']:+.2f} rests on {R['prosus_n']:,} "
           f"usable days, where a 504-day trailing window and a 126-day forward window overlap so "
           f"heavily that a handful of independent episodes carry the whole regression — the "
           f"Valkanov problem in miniature. Half-lives corroborate: Bollore {R['bol_hl']:,} "
           f"trading days against Heineken Holding's {R['heio_hl']}, no common scale at all."),

        md("## (b) The pair race\n\n"
           "The hedge ratio is not a free parameter. Since `P_holdco = (1 - d) * k * P_stake`, "
           "taking logs gives `log P_holdco = log(1 - d) + log(k * P_stake)` — so a "
           "**dollar-neutral** long/short earns exactly `Δ log(1 - d)` and nothing else. "
           "(Shorting the whole look-through stake value `k*P_stake/P_holdco` is the correct "
           "hedge only for a gap fixed in *money*; on a proportional gap it over-hedges and "
           "leaves a residual short. Reported as a variant further down.)\n\n"
           "Both arms are long/short pairs funded from collateral that itself earns cash, so "
           "**both are excess-of-cash by construction** — the race is like for like without "
           "subtracting anything. Positions are gross-normalised to one unit of notional, "
           "equal-weighted across active names, and lagged exactly one day."),
        code(
            "print('timed      net Sharpe %+.3f (HAC t %+.2f)  CAGR %+.2f%%  vol %.2f%%  maxDD %.1f%%'\n"
            "      % (R['timed_sharpe'], R['timed_t'], R['timed_cagr'], R['timed_vol'], R['timed_dd']))\n"
            "print('           gross Sharpe %+.3f (t %+.2f)   invested %.1f%% of days'\n"
            "      % (R['timed_gross'], R['timed_gross_t'], R['timed_inv']))\n"
            "print('always-on  net Sharpe %+.3f (HAC t %+.2f)  CAGR %+.2f%%  maxDD %.1f%%'\n"
            "      % (R['always_sharpe'], R['always_t'], R['always_cagr'], R['always_dd']))\n"
            "print('           gross Sharpe %+.3f (t %+.2f)'\n"
            "      % (R['always_gross'], R['always_gross_t']))\n"
            "print()\n"
            "print('timing advantage %+.3f   HAC t on the daily difference %+.2f'\n"
            "      % (R['adv'], R['t_diff']))\n"
            "print()\n"
            "print('bootstrap Sharpe CIs (2,000 draws, 21-day blocks):')\n"
            "print('  timed net    %+.3f  [%+.3f, %+.3f]  share<0 %.1f%%   <- includes zero'\n"
            "      % (R['timed_sharpe'], R['ci_net_lo'], R['ci_net_hi'], R['ci_net_neg']))\n"
            "print('  timed gross  %+.3f  [%+.3f, %+.3f]'\n"
            "      % (R['timed_gross'], R['ci_gross_lo'], R['ci_gross_hi']))\n"
            "print('  always-on    %+.3f  [%+.3f, %+.3f]  share<0 %.1f%%'\n"
            "      % (R['always_sharpe'], R['ci_always_lo'], R['ci_always_hi'], R['ci_always_neg']))"
        ),
        md(f"The gross number clears the bar (*t* = {R['timed_gross_t']:+.2f}); the net one does "
           f"not (*t* = {R['timed_t']:+.2f}, CI includes zero). The entire result lives in the gap "
           f"between gross and net — the definition of a friction-sized effect, and the reason the "
           f"next cell matters more than this one."),

        md("## The stale-quote decomposition — the check that decides it\n\n"
           "Three of the seven pairs are thin OTC ADRs in New York (NPSNY, PROSY, SFTBY). Their "
           "closing prints can be stale by hours or simply unrefreshed. In a long/short pair a "
           "stale leg **manufactures** apparent mean reversion: today's spread is measured against "
           "a price that has not moved yet, and tomorrow it \"reverts\" when the quote catches up. "
           "That is a microstructure artefact with the same signature as the effect under test, so "
           "it has to be separated out.\n\n"
           "> 💡 **In plain words** — if one of the two prices is yesterday's, the gap you measure "
           "is partly fictional, and it will look like it closes tomorrow."),
        code(
            "rows = [('full panel (7)', R['timed_gross'], R['timed_gross_t'], R['timed_sharpe'], R['timed_t']),\n"
            "        ('primary listings only (4)', R['clean_gross'], R['clean_gross_t'], R['clean_net'], R['clean_net_t']),\n"
            "        ('the 3 OTC ADR pairs alone', R['adr_gross'], R['adr_gross_t'], R['adr_net'], R['adr_net_t'])]\n"
            "print('%-28s %8s %7s %8s %7s' % ('sub-panel', 'gross', '(t)', 'net', '(t)'))\n"
            "for tag, gs, gt, ns, nt in rows:\n"
            "    print('%-28s %+8.3f %+7.2f %+8.3f %+7.2f' % (tag, gs, gt, ns, nt))\n"
            "print()\n"
            "print('pooled mean-reversion slope on the clean four: %+.4f (t %+.2f) -- still the wrong sign'\n"
            "      % (R['clean_mr_beta'], R['clean_mr_t']))\n"
            "print()\n"
            "print('per-name standalone (nobody clears |t| = 2):')\n"
            "print('  Prosus            %+.3f (t %+.2f)   <- a thin ADR' % (R['pn_prosus'], R['pn_prosus_t']))\n"
            "print('  SoftBank          %+.3f (t %+.2f)   <- a thin ADR' % (R['pn_softbank'], R['pn_softbank_t']))\n"
            "print('  Heineken Holding  %+.3f' % R['pn_heio'])\n"
            "print('  Christian Dior    %+.3f' % R['pn_dior'])\n"
            "print('  Liberty Broadband %+.3f' % R['pn_lbrd'])\n"
            "print('  Bollore           %+.3f' % R['pn_bol'])"
        ),
        md(f"Gross *t* falls **{R['timed_gross_t']:+.2f} → {R['clean_gross_t']:+.2f}** and net *t* "
           f"to **{R['clean_net_t']:+.2f}** once the stale-quote names are removed; the two "
           f"strongest standalone contributors are precisely the two thinnest ADRs. The pooled "
           f"regression on the clean four is still the wrong sign "
           f"({R['clean_mr_beta']:+.4f}, *t* = {R['clean_mr_t']:+.2f})."),

        md("## The free parameters, swept whole\n\n"
           "`enter` and `exit` in trailing-z units are the study's only genuinely free "
           "parameters — `k` and the hedge ratio are derived, costs and borrow are priced "
           "inputs — so an unswept default here would be a hidden choice. The 504-day "
           "standardisation window is a convention, and gets the same treatment. Both grids are "
           "reported whole rather than at the cell that was published."),
        code(
            "print('enter/exit grid (%d cells with a hysteresis band):' % R['thr_cells'])\n"
            "print('  gross t ranges [%+.2f, %+.2f]' % (R['thr_gross_lo'], R['thr_gross_hi']))\n"
            "print('  net   t ranges [%+.2f, %+.2f]' % (R['thr_net_lo'], R['thr_net_hi']))\n"
            "print('  cells with net t >= 2 : %d of %d   <- THE POINT'\n"
            "      % (R['thr_cells_net2'], R['thr_cells']))\n"
            "print('  headline cell (1.0/0.0) gross %+.3f (t %+.2f); best cell %+.3f (t %+.2f)'\n"
            "      % (R['timed_gross'], R['timed_gross_t'], R['thr_best_gross'], R['thr_best_gross_t']))\n"
            "print()\n"
            "print('trailing-standardisation window (headline 504d):')\n"
            "for w, t in [(252, R['zw252_t']), (378, R['zw378_t']), (504, R['timed_gross_t']),\n"
            "             (756, R['zw756_t']), (1008, R['zw1008_t'])]:\n"
            "    flag = '   <- headline, and the MAXIMUM of the sweep' if w == 504 else ''\n"
            "    print('  %4dd  gross t %+.2f%s' % (w, t, flag))"
        ),
        md(f"Two readings, and the study owes the reader both. In the threshold grid the "
           f"published cell is *not* the best one ({R['thr_best_gross']:+.3f} at a lazier exit "
           f"beats it), so the headline is not the top of a mined surface — but **no cell in the "
           f"grid reaches a net *t* of 2**, so there is no threshold pair at which this is "
           f"significant after friction. The window sweep is the less flattering of the two: "
           f"504 days is the **maximum**, and at 1008 days the gross *t* falls to "
           f"{R['zw1008_t']:+.2f} (Sharpe {R['zw1008']:+.3f}). Conclusion (a) is flat at every "
           f"window, so nothing about the mean-reversion answer turns on it; the gross trading "
           f"number — already the weaker half of the case — is worth about a point of *t* less "
           f"anywhere else."),

        md("## Sweeps and robustness"),
        code(
            "print('cost sweep (one-way bps per leg, borrow fixed at 100 bps):')\n"
            "for c, sh, t in [(0, R['cost0'], R['cost0_t']), (5, R['cost5'], R['cost5_t']),\n"
            "                 (10, R['timed_sharpe'], R['timed_t']),\n"
            "                 (25, R['cost25'], R['cost25_t']), (50, R['cost50'], R['cost50_t'])]:\n"
            "    print('  %2d bps  Sharpe %+.3f (t %+.2f)' % (c, sh, t))\n"
            "print()\n"
            "print('borrow sweep (annualised bps on the short leg, cost fixed at 10 bps):')\n"
            "for b, sh, t in [(0, R['borrow0'], R['borrow0_t']),\n"
            "                 (100, R['timed_sharpe'], R['timed_t']),\n"
            "                 (600, R['borrow600'], R['borrow600_t'])]:\n"
            "    print('  %3d bps  Sharpe %+.3f (t %+.2f)' % (b, sh, t))\n"
            "print()\n"
            "print('era cut (2016): early %+.3f (t %+.2f)   late %+.3f (t %+.2f)'\n"
            "      % (R['era_e_sh'], R['era_e_t'], R['era_l_sh'], R['era_l_t']))\n"
            "print('leave-one-out : timed Sharpe stays in [%+.3f, %+.3f], t in [%+.2f, %+.2f]'\n"
            "      % (R['loo_lo'], R['loo_hi'], R['loo_t_lo'], R['loo_t_hi']))\n"
            "print('NAV-calibration sweep (k x0.8..1.2, other x0.5..1.5): Sharpe in [%+.3f, %+.3f]'\n"
            "      % (R['calib_lo'], R['calib_hi']))\n"
            "print('look-through hedge variant: net %+.3f (t %+.2f), gross %+.3f'\n"
            "      % (R['lt_net'], R['lt_net_t'], R['lt_gross']))\n"
            "print()\n"
            "print('long-only, excess-of-cash: timed %+.3f | hold all holdcos %+.3f | hold the stakes %+.3f'\n"
            "      % (R['lo_timed'], R['lo_holdcos'], R['lo_stakes']))\n"
            "print('  HAC t on (timed - hold all holdcos): %+.2f  -> unhedged timing adds nothing'\n"
            "      % R['lo_t_diff'])\n"
            "print('cash-leg cross-check: ^IRX proxy %.2f%%/yr vs BIL total return %.2f%%/yr'\n"
            "      % (R['irx_cagr'], R['bil_cagr']))"
        ),
        md(f"The NAV-calibration sweep clears the study's own biggest liability: scaling `k` from "
           f"0.8× to 1.2× and the `other` term from 0.5× to 1.5× moves the timed Sharpe only "
           f"between {R['calib_lo']:+.3f} and {R['calib_hi']:+.3f}, and the pooled mean-reversion "
           f"*t* stays inside ±1.2 throughout. The trailing-z design did its job — the flat answer "
           f"is not an artefact of the assumption table.\n\n"
           f"The look-through hedge variant turns negative ({R['lt_net']:+.3f}), but that is "
           f"mechanical rather than informative: over-hedging a proportional gap leaves the pair "
           f"net short the underlying through a 22-year bull market, so that number is about beta. "
           f"It is reported because it shows how construction-sensitive this corner is."),

        md("## Live synthetic control — the machinery is unbiased\n\n"
           "A panel of synthetic (stake, holdco) pairs where the discount lives in **logit space**, "
           "so it stays inside (0,1) without a reflecting barrier, plus an Itô correction that "
           "makes the discount itself a **martingale** when the mean-reversion pull is switched "
           "off. Both details matter: a barrier, or an uncorrected logistic drift, would have "
           "smuggled reversion into the very control that is supposed to have none. Planted OU "
           "must fire; the martingale null must not."),
        code(
            SYN_SETUP +
            "import numpy as np\n"
            "for ss, tag in [(1.0, 'planted OU     '), (0.0, 'martingale null')]:\n"
            "    rows = [st.synthetic_detect(data.synthetic_panel(signal_strength=ss, seed=957 + 37 * s)[0])\n"
            "            for s in range(3)]\n"
            "    t = np.array([r['pooled_t'] for r in rows])\n"
            "    g = np.array([r['sharpe_timed_gross'] for r in rows])\n"
            "    print('%s: pooled beta t mean %+6.2f   gross timed Sharpe mean %+.3f'\n"
            "          % (tag, t.mean(), g.mean()))"
        ),
        md(f"On the frozen four-seed run in `docs/results.md` the planted world gives a pooled *t* "
           f"of **{R['syn_pl_t']:+.2f}** and a gross Sharpe of **{R['syn_pl_gross']:+.3f}**; the "
           f"null gives **{R['syn_nl_t']:+.2f}** and **{R['syn_nl_gross']:+.3f}**. The detector "
           f"finds reversion when it is planted and is quiet when it is not, so the flat real-tape "
           f"answer is a property of holdcos and not of the harness."),

        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The load-bearing half of the thesis is absent: pooled "
           f"Driscoll-Kraay slope **{R['pooled_beta']:+.4f}** (*t* = **{R['pooled_t']:+.2f}**, "
           f"R² = {R['pooled_r2']:.4f}), the right sign in **{R['n_neg']}/7** names — a coin "
           f"flip — and two names where a wide gap got wider. The trading half posts a gross Sharpe of "
           f"**{R['timed_gross']:+.3f}** (*t* = **{R['timed_gross_t']:+.2f}**) — but it falls to "
           f"*t* = {R['timed_t']:+.2f} at 10 bps a leg, with a bootstrap CI of "
           f"[{R['ci_net_lo']:+.3f}, {R['ci_net_hi']:+.3f}] that includes zero, and it falls to "
           f"gross *t* = **{R['clean_gross_t']:+.2f}** / net **{R['clean_net_t']:+.2f}** once the "
           f"three thin OTC ADR pairs are removed. Sweep the two thresholds that define the rule "
           f"and **{R['thr_cells_net2']} of {R['thr_cells']}** cells reach a net *t* of 2; sweep "
           f"the standardisation window and the headline turns out to be the best of five. No "
           f"single name clears |*t*| = 2 standalone; neither era does. What survives is a "
           f"friction-sized reversal concentrated in the least liquid corner of the panel.\n"
           f"- **Tradability — Mirage.** Flat at 25 bps one-way ({R['cost25']:+.3f}), "
           f"{R['cost50']:+.3f} at 50; {R['borrow600']:+.3f} at a 6% borrow; a net CI spanning "
           f"zero; sign-flipping under the alternative hedge; and a required short leg in exactly "
           f"the illiquid ADRs producing the signal. The patient version — own the discount and "
           f"wait — returned **{R['always_cagr']:+.2f}%/yr** over 22 years.\n"
           f"- **Survivorship, named.** The panel contains only holdcos that still exist and whose "
           f"stake is still listed. Liberty TripAdvisor and Cannae, both classic wide-discount "
           f"names, were dropped because their 2025 take-privates left no usable history — and a "
           f"discount that ends in a buyout is a discount that *closed*. The bias therefore runs "
           f"**in favour** of the thesis, and the thesis still failed."),
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
