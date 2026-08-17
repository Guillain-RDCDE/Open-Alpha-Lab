"""Generate the two narrative notebooks for Study 913 (Tracking-Difference Persistence).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministically. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the fast
offline synthetic control, and they are labelled synthetic wherever they appear. No cell
falls back to synthetic data under a real-tape banner.

Generated code cells are assembled by plain concatenation of ``repr`` literals — never by
``%``-formatting a template — so that format specifiers inside the generated code cannot be
consumed at build time.
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
# Frozen real-tape headline — mirror of docs/results.md. Daily total-return closes,
# auto_adjust=True, as-of 2026-06-30, complete calendar years only, 1 bp one-way cost.
# --------------------------------------------------------------------------- #
R = dict(
    asof="2026-06-30",
    trio_start="2010-09-09", trio_end="2026-06-30", trio_n=3975, trio_fp="f11699949902",
    trio_years=15,
    lad_start="2011-05-04", lad_end="2026-06-30", lad_n=3811, lad_fp="82068f45e436",
    lad_years=14,
    ndx_start="2020-10-13", ndx_end="2026-06-30", ndx_n=1434, ndx_fp="b33e0709c240",
    ndx_years=5,
    # the measurement floor vs the fee ladder
    trio_td_sd=10.7, lad_td_sd=9.7, ndx_td_sd=5.0,
    trio_fee_spread=6.45, lad_fee_spread=7.95, ndx_fee_spread=5.0,
    # persistence
    trio_rho=0.071, trio_rho_t=0.43, trio_pairs=14, trio_perm_p=0.964,
    trio_rho_res=-0.179, trio_rho_res_t=-0.92, trio_rho_allpairs=0.305,
    lad_rho=0.253, lad_rho_t=2.34, lad_pairs=13, lad_perm_p=0.307,
    lad_rho_res=0.200, lad_rho_res_t=1.71, lad_rho_allpairs=0.195,
    ndx_sign_run="5/5", ndx_pairs=4,
    # the rules, 1 bp one-way
    trio_win_cheap=0.33, trio_win_cheap_t=0.10, trio_win_cheap_pos="7/14",
    trio_cheap_lead=3.27, trio_cheap_lead_t=1.26, trio_cheap_lead_pos="12/14",
    lad_win_cheap=-6.08, lad_win_cheap_t=-1.03, lad_win_cheap_pos="3/13",
    lad_cheap_lead=10.64, lad_cheap_lead_t=4.85, lad_cheap_lead_pos="11/13",
    ndx_cheap_lead=8.79, ndx_cheap_lead_t=2.93, ndx_cheap_lead_pos="4/4",
    # annual HAC (Newey-West, 1 annual lag) on the same per-year gap series
    trio_win_cheap_hac=0.10, trio_cheap_lead_hac=1.40,
    lad_win_cheap_hac=-1.22, lad_cheap_lead_hac=4.78, ndx_cheap_lead_hac=4.32,
    # the hindsight-free control: no fee sheet at all, buy-and-hold vs the flagship
    lad_ew_gap=5.21, lad_ew_t=2.68, lad_ew_hac=2.77, lad_ew_pos="11/14",
    lad_ew_ci=(1.8, 8.8),
    ndx_raw_gap=8.55, ndx_raw_t=3.95, ndx_raw_pos="5/5", ndx_raw_ci=(4.7, 12.5),
    tie_rebal_freebie=0.06,
    lad_cheap_lead_ci=(6.1, 14.8), lad_win_cheap_ci=(-17.4, 0.9),
    ndx_cheap_lead_ci=(4.5, 13.1), trio_win_cheap_ci=(-5.9, 6.6),
    trio_switches=10, trio_live_years=14, lad_switches=8, lad_live_years=13,
    # daily unit (the conservative floor)
    trio_win_cheap_daily_t=0.03, trio_cheap_lead_daily_t=0.52,
    lad_win_cheap_daily_t=-0.89, lad_cheap_lead_daily_t=1.32,
    ndx_cheap_lead_daily_t=0.28,
    # eras
    trio_era_e="2012-2018", trio_era_e_gap=-0.55, trio_era_e_t=-0.12,
    trio_era_l="2019-2025", trio_era_l_gap=7.09, trio_era_l_t=4.02,
    lad_era_e="2013-2018", lad_era_e_gap=10.94, lad_era_e_t=3.52,
    lad_era_l="2019-2025", lad_era_l_gap=10.39, lad_era_l_t=3.14,
    # synthetic control (8 seeds each) — machinery proof only
    syn_pl_rho=0.725, syn_pl_gap=35.66, syn_nl_rho=-0.079, syn_nl_gap=-0.29,
)

# Excess-of-cash Sharpe vs BIL, S&P 500 ETF trio (both arms excess).
SHARPE = [("winner", 0.8230), ("cheapest", 0.8228), ("leader", 0.8241), ("eqw", 0.8234)]

# Relative TD for the S&P 500 ETF trio, bp vs the family mean (frozen real tape).
TRIO_TD = [
    (2011, 0.5, -1.2, 0.7), (2012, -2.3, 4.4, -2.1), (2013, -2.6, -2.7, 5.3),
    (2014, 14.4, 24.8, -39.2), (2015, -4.7, 0.3, 4.4), (2016, 9.6, -36.1, 26.5),
    (2017, -3.9, 1.0, 2.9), (2018, -4.9, 3.0, 1.9), (2019, -5.6, -2.8, 8.5),
    (2020, -2.0, 4.6, -2.7), (2021, -3.2, -0.2, 3.4), (2022, -0.5, 0.9, -0.4),
    (2023, -9.3, 3.7, 5.5), (2024, -4.6, 0.2, 4.4), (2025, -7.7, 5.0, 2.7),
]

# Cost sweep, winner - cheapest, S&P 500 ETF trio: (one-way bp, gap bp/yr, annual t).
COST_GRID = [(0.0, 2.01, 0.60), (1.0, 0.33, 0.10), (5.0, -6.39, -1.77),
             (10.0, -14.79, -3.35), (25.0, -39.98, -4.92)]

# Mean TD vs the PRICE-ONLY ^GSPC proxy, bp/yr — i.e. the dividend yield.
GSPC_TD = [("SPY", 201), ("IVV", 203), ("VOO", 204),
           ("VFIAX", 207), ("FXAIX", 211), ("SWPPX", 205)]

# Tax ASSUMPTION grid: (embedded gain %, years to repay at 15% / 20% / 23.8%).
TAX_GRID = [(10, 23, 31, 37), (25, 58, 78, 92), (50, 116, 155, 185), (100, 233, 310, 369)]

# The measurement floor: (family, sd of relative TD, fee spread contained, complete years).
FLOOR = [("S&P 500 ETF trio", R["trio_td_sd"], R["trio_fee_spread"], R["trio_years"]),
         ("S&P 500 full ladder", R["lad_td_sd"], R["lad_fee_spread"], R["lad_years"]),
         ("Nasdaq-100 pair", R["ndx_td_sd"], R["ndx_fee_spread"], R["ndx_years"])]

# Persistence table: (family, rho(y,y+1), t, pairs, permutation p, rho all pairs, residual).
PERSIST = [
    ("S&P 500 ETF trio", R["trio_rho"], R["trio_rho_t"], R["trio_pairs"],
     R["trio_perm_p"], R["trio_rho_allpairs"], R["trio_rho_res"]),
    ("S&P 500 full ladder", R["lad_rho"], R["lad_rho_t"], R["lad_pairs"],
     R["lad_perm_p"], R["lad_rho_allpairs"], R["lad_rho_res"]),
]

# Rule gaps: (family, gap name, bp/yr, annual t, annual HAC t, daily HAC t, years positive).
GAPS = [
    ("trio", "winner-cheapest", R["trio_win_cheap"], R["trio_win_cheap_t"],
     R["trio_win_cheap_hac"], R["trio_win_cheap_daily_t"], R["trio_win_cheap_pos"]),
    ("trio", "cheapest-leader", R["trio_cheap_lead"], R["trio_cheap_lead_t"],
     R["trio_cheap_lead_hac"], R["trio_cheap_lead_daily_t"], R["trio_cheap_lead_pos"]),
    ("ladder", "winner-cheapest", R["lad_win_cheap"], R["lad_win_cheap_t"],
     R["lad_win_cheap_hac"], R["lad_win_cheap_daily_t"], R["lad_win_cheap_pos"]),
    ("ladder", "cheapest-leader", R["lad_cheap_lead"], R["lad_cheap_lead_t"],
     R["lad_cheap_lead_hac"], R["lad_cheap_lead_daily_t"], R["lad_cheap_lead_pos"]),
    ("ndx", "cheapest-leader", R["ndx_cheap_lead"], R["ndx_cheap_lead_t"],
     R["ndx_cheap_lead_hac"], R["ndx_cheap_lead_daily_t"], R["ndx_cheap_lead_pos"]),
]

# The hindsight-free control: each fund vs its family flagship, buy-and-hold, NO fee sheet.
# (fund-minus-flagship label, bp/yr, annual t, years positive, complete years)
HINDSIGHT_FREE = [
    ("IVV - SPY", 2.37, 0.62, "12/14"),
    ("VOO - SPY", 3.46, 0.75, "12/14"),
    ("SWPPX - SPY", 4.47, 1.59, "10/14"),
    ("VFIAX - SPY", 6.04, 2.49, "10/14"),
    ("FXAIX - SPY", 9.71, 4.38, "11/14"),
    ("EQW of all five - SPY", R["lad_ew_gap"], R["lad_ew_t"], R["lad_ew_pos"]),
    ("QQQM - QQQ", R["ndx_raw_gap"], R["ndx_raw_t"], R["ndx_raw_pos"]),
]

# Era table: (label, early span, early gap, early t, late span, late gap, late t).
ERAS = [
    ("trio   cheapest-leader", R["trio_era_e"], R["trio_era_e_gap"], R["trio_era_e_t"],
     R["trio_era_l"], R["trio_era_l_gap"], R["trio_era_l_t"]),
    ("ladder cheapest-leader", R["lad_era_e"], R["lad_era_e_gap"], R["lad_era_e_t"],
     R["lad_era_l"], R["lad_era_l_gap"], R["lad_era_l_t"]),
]

SETUP = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
)

INTRO = f"""**Two funds track the same index. One quietly returns a few basis points more
each year. Is last year's winner next year's winner?**

The fund-picker's folklore says yes: look up last year's **tracking difference** — the gap
between a fund's total return and its index's — and buy whoever won it, because good
tracking is a skill and skills persist. The duller rival rule says just buy the lowest
**expense ratio**, which is published in advance and costs nothing to read.

We test both on two families of daily **total-return** closes, as-of {R['asof']}:

- **S&P 500 ETF trio** — SPY (9.45 bp), IVV (3 bp), VOO (3 bp): {R['trio_start']} →
  {R['trio_end']}, {R['trio_n']:,} days, {R['trio_years']} complete calendar years
- **S&P 500 full ladder** — the trio plus three NAV-priced index mutual funds
  (VFIAX 4 bp, FXAIX 1.5 bp, SWPPX 2 bp), which widen the fee ladder
- **Nasdaq-100 pair** — QQQ (20 bp) and QQQM (15 bp), {R['ndx_years']} complete years

*Numbers below are the frozen headline run (`docs/results.md`, fingerprints
`{R['trio_fp']}` / `{R['lad_fp']}` / `{R['ndx_fp']}`); the only live cells run the offline
synthetic control and say so. **SPLG was requested and is unavailable** — Yahoo! Finance
serves a single stale bar for it, so it is declared missing rather than quietly swapped.*
"""

HEADER_CURIOUS = "# Study 913 — Tracking-Difference Persistence 🧾\n\n" + INTRO
HEADER_QUANTS = ("# Study 913 — Tracking-Difference Persistence 🧾\n\n"
                 "*The quant teardown. The plain-language version is "
                 "[01_for_the_curious](01_for_the_curious.ipynb).*\n\n" + INTRO)


# --------------------------------------------------------------------------- #
# 01 — for the curious
# --------------------------------------------------------------------------- #
def build_curious():
    cells = [
        md(HEADER_CURIOUS),

        md("## 1. What a tracking difference actually is\n\n"
           "An index fund promises to hand you the index. It never quite does. The fee comes "
           "out, dividends get reinvested a day late, the fund trades to keep up when the index "
           "reshuffles, and some of the shortfall comes back as securities-lending income. The "
           "annual leftover is the **tracking difference** — usually a handful of basis points, "
           "where a basis point is one hundredth of one percent.\n\n"
           "So: does the fund that tracked best last year track best again? Here is the real "
           "table for the three big S&P 500 ETFs, each year's returns measured against the "
           "average of the three."),

        code(
            "TRIO = " + repr(TRIO_TD) + "\n"
            "print('Relative tracking difference, S&P 500 ETF trio (bp vs the family average)')\n"
            "header = 'year'.rjust(6) + 'SPY'.rjust(9) + 'IVV'.rjust(9) + 'VOO'.rjust(9)\n"
            "print(header + '   best that year')\n"
            "for year, spy, ivv, voo in TRIO:\n"
            "    row = {'SPY': spy, 'IVV': ivv, 'VOO': voo}\n"
            "    best = max(row, key=row.get)\n"
            "    line = str(year).rjust(6)\n"
            "    for v in (spy, ivv, voo):\n"
            "        line += format(v, '+.1f').rjust(9)\n"
            "    print(line + '   ' + best)\n"
        ),

        md(f"## 2. Look at 2014 and 2016 before believing any of it\n\n"
           f"VOO 'lost' **39 bp** in 2014 and IVV 'lost' **36 bp** in 2016 — and both were back "
           f"to normal the very next year. Those are not 40 basis points of real slippage; they "
           f"are artefacts of how the data vendor time-stamps a dividend.\n\n"
           f"That matters enormously. The year-to-year wobble in this table has a standard "
           f"deviation of **{R['trio_td_sd']:.1f} bp**, while the actual fee gap the trio "
           f"contains is only **{R['trio_fee_spread']:.2f} bp** (SPY charges 9.45, IVV and VOO "
           f"charge 3). **The thing we are trying to see is smaller than the measurement error.** "
           f"Whatever last year's ranking says, most of it is noise.\n\n"
           "> 🔬 **For the quants** — the identification problem in one line: the estimator's "
           "per-year standard error exceeds the cross-sectional spread of the parameter. No "
           "amount of cleverness recovers a 6 bp ladder from a 10 bp floor in a single annual "
           "observation."),

        md(f"## 3. So does last year's winner repeat?\n\n"
           f"Rank the three ETFs each year and ask how strongly the ranking carries over. It "
           f"barely does: the average year-over-year rank correlation is **{R['trio_rho']:+.3f}** "
           f"across {R['trio_pairs']} year-pairs — indistinguishable from a coin flip "
           f"(*t* = {R['trio_rho_t']:+.2f}; shuffling the years into a random order reproduces it "
           f"{R['trio_perm_p'] * 100:.0f}% of the time).\n\n"
           f"And the rule built on it earns exactly what you would expect."),

        code(
            "win_cheap, win_cheap_t, win_cheap_pos = " + repr(
                (R["trio_win_cheap"], R["trio_win_cheap_t"], R["trio_win_cheap_pos"])) + "\n"
            "cheap_lead, cheap_lead_t, cheap_lead_pos = " + repr(
                (R["trio_cheap_lead"], R["trio_cheap_lead_t"], R["trio_cheap_lead_pos"])) + "\n"
            "switches, live_years = " + repr((R["trio_switches"], R["trio_live_years"])) + "\n"
            "print('S&P 500 ETF trio — annual rebalance, one-day lag, 1 bp one-way cost')\n"
            "print()\n"
            "print('  hold last year\\'s winner  vs  hold the cheapest fund : '\n"
            "      + format(win_cheap, '+.2f') + ' bp/yr  (t = ' + format(win_cheap_t, '+.2f')\n"
            "      + ', positive in ' + win_cheap_pos + ' years)')\n"
            "print('  hold the cheapest fund   vs  hold the flagship     : '\n"
            "      + format(cheap_lead, '+.2f') + ' bp/yr  (t = ' + format(cheap_lead_t, '+.2f')\n"
            "      + ', positive in ' + cheap_lead_pos + ' years)')\n"
            "print()\n"
            "print('  the winner rule changed fund ' + str(switches) + ' times in '\n"
            "      + str(live_years) + ' years, to buy essentially nothing.')\n"
            "print('  the cheapest rule never switched fund at all.')\n"
        ),

        md(f"## 4. Widen the fee gap and something *does* appear\n\n"
           f"Add three index **mutual funds** — Vanguard's VFIAX (4 bp), Fidelity's FXAIX "
           f"(1.5 bp) and Schwab's SWPPX (2 bp) — and the ladder becomes wide enough that "
           f"averaging over thirteen years pulls it out of the noise. Now the ranking persists "
           f"({R['lad_rho']:+.3f}, "
           f"*t* = {R['lad_rho_t']:+.2f}), and the cheapest fund beats the flagship by "
           f"**{R['lad_cheap_lead']:+.2f} bp a year** (*t* = {R['lad_cheap_lead_t']:+.2f}, "
           f"positive in {R['lad_cheap_lead_pos']} years).\n\n"
           f"The Nasdaq pair makes the same point with only two funds: **QQQM (15 bp) beat QQQ "
           f"(20 bp) in {R['ndx_sign_run']} complete years**, by {R['ndx_raw_gap']:+.2f} bp a "
           f"year on average ({R['ndx_cheap_lead']:+.2f} over the four years a rule could "
           f"actually have traded it — the first year always goes to forming the ranking).\n\n"
           f"**Two honest deductions from that {R['lad_cheap_lead']:+.2f}.** It is measured with "
           f"the fee sheet published *today*: Fidelity only cut FXAIX to 1.5 bp in 2019, so 'the "
           f"cheapest fund' is partly chosen after the fact. Buy *all five* non-flagship funds "
           f"equally instead — no fee sheet, no ranking, no forecast — and the gap is "
           f"{R['lad_ew_gap']:+.2f} bp/yr (*t* = {R['lad_ew_t']:+.2f}), about half. And the "
           f"cheapest-vs-flagship number is a *level*, not a memory: shuffle the calendar years "
           f"into a random order and the same 'persistence' survives — because what persists is "
           f"not last year's *result*, it is the fee, a fixed number printed on the fund's own "
           f"website, in advance, for free. You never needed last year's tape."),

        md(f"## 5. The rule that works is not a strategy\n\n"
           f"The honest summary is: **buy a cheap share class, and then never touch it.** That is "
           f"worth {R['trio_cheap_lead']:.1f} to {R['lad_ew_gap']:.1f} bp a year with zero "
           f"turnover — genuinely real, and genuinely a purchase decision rather than an edge. "
           f"(The {R['lad_cheap_lead']:.1f} bp version needs you to know which fund would end up "
           f"cheapest; {R['lad_ew_gap']:.1f} bp is what you could have had knowing only that SPY "
           f"was the dearest.)\n\n"
           f"It will also not justify moving money you already have, because switching an "
           f"appreciated holding realises the capital gain. At the published "
           f"{R['trio_fee_spread']:.2f} bp SPY→VOO gap, here is how long the cheaper fee takes to "
           f"repay that one-off tax bill."),

        code(
            "TAX = " + repr(TAX_GRID) + "\n"
            "print('Years for the 6.45 bp/yr fee gap to repay the tax on a SPY -> VOO switch')\n"
            "print('(an ASSUMPTION grid — neither the gain nor the tax rate is on the tape)')\n"
            "print()\n"
            "print('embedded gain'.rjust(14) + 'sheltered'.rjust(12) + '15%'.rjust(8)\n"
            "      + '20%'.rjust(8) + '23.8%'.rjust(8))\n"
            "for gain, t15, t20, t238 in TAX:\n"
            "    line = (str(gain) + '%').rjust(14) + '0'.rjust(12)\n"
            "    for v in (t15, t20, t238):\n"
            "        line += str(v).rjust(8)\n"
            "    print(line)\n"
            "print()\n"
            "print('In a tax-sheltered account the switch is free.')\n"
            "print('In a taxable one with a +50% gain at 20%, it takes 155 years to pay for itself.')\n"
        ),

        md("## 6. Live check — the machinery is not broken (offline **synthetic** data)\n\n"
           "Before believing a null result, check that the detector fires when there really is "
           "something to find. The cell below runs on a **synthetic** panel of index funds, not "
           "the real tape: one world has a genuine 30 bp fee ladder, the other has identical "
           "fees. The pipeline must find persistence in the first and none in the second."),

        code(
            SETUP +
            "import numpy as np\n"
            "from td_persist import data, strategy as st\n"
            "for label, ss in [('planted 30 bp fee ladder', 1.0), ('flat-fee null           ', 0.0)]:\n"
            "    rho, gap = [], []\n"
            "    for s in range(8):\n"
            "        px, truth = data.synthetic_panel(signal_strength=ss, seed=913 + s)\n"
            "        d = st.synthetic_detect(px, truth)\n"
            "        rho.append(d['mean_spearman'])\n"
            "        gap.append(d['cheapest_minus_leader_bp'])\n"
            "    print(label + ' (SYNTHETIC, 8 seeds): rank persistence '\n"
            "          + format(np.mean(rho), '+.3f') + '  |  cheapest-minus-dearest '\n"
            "          + format(np.mean(gap), '+.2f') + ' bp/yr')\n"
        ),

        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The claim as posed fails exactly where it would be useful: "
           f"among near-identical S&P 500 ETFs, last year's tracking-difference rank says nothing "
           f"about next year's ({R['trio_rho']:+.3f}, *t* = {R['trio_rho_t']:+.2f}), because a "
           f"{R['trio_fee_spread']:.2f} bp fee gap cannot be read through a "
           f"{R['trio_td_sd']:.1f} bp measurement floor. What *is* real is the thing underneath: "
           f"across a wider ladder the cheapest fund beats the flagship by "
           f"{R['lad_cheap_lead']:+.2f} bp/yr (*t* = {R['lad_cheap_lead_t']:+.2f}) — "
           f"{R['lad_ew_gap']:+.2f} (*t* = {R['lad_ew_t']:+.2f}) once you are not allowed to pick "
           f"the cheapest fund with hindsight — and QQQM beat QQQ in {R['ndx_sign_run']} years. "
           f"But that is a published *level*, not a memory — the ranking is the fee sheet, "
           f"available in advance.\n"
           f"- **Tradability — Fragile.** The bankable version is a one-off purchase decision "
           f"worth a few basis points with no turnover. The rotation version earns "
           f"{R['trio_win_cheap']:+.2f} bp/yr for {R['trio_switches']} trades in "
           f"{R['trio_live_years']} years, turns negative at any realistic switching cost, and is "
           f"far too thin to move an existing taxable holding."),
    ]
    nb = new_notebook()
    nb["cells"] = cells
    return nb


# --------------------------------------------------------------------------- #
# 02 — for the quants
# --------------------------------------------------------------------------- #
def build_quants():
    cells = [
        md(HEADER_QUANTS),

        md("## Design\n\n"
           "**Estimand.** For fund *f* in complete calendar year *y*, the relative tracking "
           "difference `TD[f, y] = r[f, y] − mean_g r[g, y]`, in bp, where `r` is the compounded "
           "total return of the adjusted close. A common per-year constant shifts every fund "
           "equally, so **ranks are invariant** to the choice of index proxy (family mean, family "
           "leader, or any member); only the reported level moves — and the level is not "
           "identified from a public tape at all.\n\n"
           "**Sample discipline.** Complete calendar years only (≥ 200 trading days), at both "
           "ends. QQQM listed 2020-10-13, so 2020 is a stub and is dropped; the as-of is "
           "2026-06-30, so 2026 is dropped. Year-pairs are never chained across a dropped year.\n\n"
           "**Execution.** Exactly one lag: the ranking uses returns through the last close of "
           "year *y*; the weight vector is broadcast over year *y+1*'s days and shifted forward "
           "one trading day, so it first earns on the **second** trading day of *y+1*. Costs are "
           "one-way × NAV on realised turnover. No short leg, so no borrow.\n\n"
           "**Arms.** `winner` = argmax TD[·, y−1]; `loser` = argmin; `cheapest` = argmin "
           "published expense ratio, equal-split across ties, never traded; `leader` = the "
           "flagship, never traded; `eqw`.\n\n"
           "> 💡 **In plain words** — rank the funds on last year's shortfall, buy the best one "
           "on the second trading day of January, pay a spread, and see whether you beat someone "
           "who bought the cheapest fund once and went to sleep."),

        md("## The proxy problem, demonstrated rather than asserted\n\n"
           "^GSPC is a price-only index. Measured against it, every S&P 500 fund shows a "
           "'tracking difference' of roughly +200 bp/yr — which is the dividend yield. The "
           "**level** of TD is not identified from a public tape. Note, though, that the "
           "*spread* of that column is exactly the fee ladder: the whole result, in disguise."),

        code(
            "GSPC = " + repr(GSPC_TD) + "\n"
            "print('TD vs ^GSPC (PRICE-ONLY proxy), bp/yr — this is the dividend yield:')\n"
            "for name, v in GSPC:\n"
            "    print('  ' + name.rjust(6) + '  ' + format(v, '+d').rjust(6) + ' bp')\n"
            "vals = [v for _, v in GSPC]\n"
            "print('  spread across funds: ' + str(max(vals) - min(vals))\n"
            "      + ' bp  <- THIS is the fee ladder; the level is not.')\n"
        ),

        md("## The measurement floor\n\n"
           "Standard deviation of relative TD against the fee spread the family actually "
           "contains, and then the same comparison against the standard error of the sample "
           "mean (sd / √years). The point is not that the ladder is invisible — it is that it "
           "is invisible **in any one year** and only emerges from averaging. A rule that reads "
           "last year's ranking is working at the left-hand column; a rule that reads the fee "
           "sheet skips the estimation problem entirely.\n\n"
           "The two ~40 bp 'outlier' years in the trio (VOO 2014, IVV 2016) fully reverse the "
           "next year and are dividend-timing artefacts of the adjusted close, not slippage."),

        code(
            "FLOOR = " + repr(FLOOR) + "\n"
            "print('family'.rjust(22) + 'sd(TD)'.rjust(10) + 'fee spread'.rjust(12)\n"
            "      + 'one year?'.rjust(11) + 'se of mean'.rjust(12) + 'full sample?'.rjust(13))\n"
            "for name, sd, spread, years in FLOOR:\n"
            "    se = sd / years ** 0.5\n"
            "    print(name.rjust(22) + (format(sd, '.1f') + ' bp').rjust(10)\n"
            "          + (format(spread, '.2f') + ' bp').rjust(12)\n"
            "          + ('yes' if spread > sd else 'NO').rjust(11)\n"
            "          + (format(se, '.2f') + ' bp').rjust(12)\n"
            "          + ('yes' if spread > se else 'NO').rjust(13))\n"
        ),

        md("## Persistence: rank correlation, a *t*, and a permutation null\n\n"
           "The *t* is a one-sample *t* on the mean of the per-pair Spearman coefficients "
           "(n = year-pairs; there is no daily autocorrelation to correct here, so no HAC). The "
           "permutation null shuffles the **order of the calendar years**, preserving each "
           "year's cross-section and destroying only the time linkage.\n\n"
           "> 💡 **In plain words** — the permutation asks whether the link is specifically "
           "between *consecutive* years, or whether any year predicts any other equally well. If "
           "the latter, it is a constant — and a constant you can look up is not a forecast."),

        code(
            "P = " + repr(PERSIST) + "\n"
            "print('family'.rjust(22) + 'rho(y,y+1)'.rjust(12) + 't'.rjust(8)\n"
            "      + 'pairs'.rjust(7) + 'perm p'.rjust(9) + 'rho(all pairs)'.rjust(16)\n"
            "      + 'residual rho'.rjust(14))\n"
            "for name, rho, t, pairs, pp, allp, res in P:\n"
            "    print(name.rjust(22) + format(rho, '+.3f').rjust(12) + format(t, '+.2f').rjust(8)\n"
            "          + str(pairs).rjust(7) + format(pp, '.3f').rjust(9)\n"
            "          + format(allp, '+.3f').rjust(16) + format(res, '+.3f').rjust(14))\n"
            "print()\n"
            "print('Trio  : no persistence at all (permutation p = 0.96).')\n"
            "print('Ladder: rho = +0.253 (t = +2.34) BUT permutation p = 0.31, and the all-pairs')\n"
            "print('        rank correlation (+0.195) matches the consecutive-pair one (+0.253).')\n"
            "print('     => a time-invariant LEVEL (the fee sheet), not a year-to-year memory.')\n"
        ),

        md(f"Two caveats on that table. Fund-demeaning over 13–14 observations imposes a known "
           f"negative small-sample bias on the residual autocorrelation, so a residual near zero "
           f"is the expected reading under 'fees and nothing else'; the ladder's "
           f"{R['lad_rho_res']:+.3f} (*t* = {R['lad_rho_res_t']:+.2f}) does not clear the bar. And "
           f"with two funds (QQQ/QQQM) Spearman degenerates to ±1, so the Nasdaq result is a "
           f"**{R['ndx_sign_run']} sign run** and is reported as one, not as a rank test."),

        md("## The rules, in two units\n\n"
           "`t (annual)` is the paired *t* on the per-year gap; `t (ann HAC)` is Newey-West with "
           "one annual lag on the same series. Calendar years do not overlap, so the two agree — "
           "the HAC column is there so independence is never simply assumed."),

        code(
            "G = " + repr(GAPS) + "\n"
            "print('family'.rjust(8) + 'gap'.rjust(18) + 'bp/yr'.rjust(10)\n"
            "      + 't (annual)'.rjust(12) + 't (ann HAC)'.rjust(13)\n"
            "      + 't (daily HAC)'.rjust(15) + 'yrs +'.rjust(8))\n"
            "for fam, gap, bp, t_ann, t_hac, t_day, pos in G:\n"
            "    print(fam.rjust(8) + gap.rjust(18) + format(bp, '+.2f').rjust(10)\n"
            "          + format(t_ann, '+.2f').rjust(12) + format(t_hac, '+.2f').rjust(13)\n"
            "          + format(t_day, '+.2f').rjust(15) + pos.rjust(8))\n"
            "print()\n"
            "print('The rule window is one year shorter than the measurement window: the first')\n"
            "print('complete year is consumed by the ranking. The Nasdaq 5/5 raw-TD run is a 4/4')\n"
            "print('once it has to be traded, and its t = +2.93 is computed on those four years.')\n"
        ),

        md(f"**Why the two units disagree, and which to believe.** The daily HAC *t* is near zero "
           f"everywhere — *including* for gaps that are strongly significant annually. That is a "
           f"power problem, not a contradiction: the day-to-day difference between two funds "
           f"holding the same basket is dominated by print timing and dividend ex-date offsets, "
           f"noise that reverses within days and swamps a 10 bp/yr drift. The estimand is defined "
           f"once a year, so the annual paired test is its natural unit; the daily HAC figure is "
           f"reported as the conservative floor rather than suppressed.\n\n"
           f"Block-bootstrap 95% CIs on the annual gap (5,000 draws, 2-year blocks): ladder "
           f"cheapest−leader **[{R['lad_cheap_lead_ci'][0]:+.1f}, "
           f"{R['lad_cheap_lead_ci'][1]:+.1f}]** and Nasdaq **[{R['ndx_cheap_lead_ci'][0]:+.1f}, "
           f"{R['ndx_cheap_lead_ci'][1]:+.1f}]** are clear of zero; ladder winner−cheapest "
           f"**[{R['lad_win_cheap_ci'][0]:+.1f}, {R['lad_win_cheap_ci'][1]:+.1f}]** and trio "
           f"winner−cheapest **[{R['trio_win_cheap_ci'][0]:+.1f}, "
           f"{R['trio_win_cheap_ci'][1]:+.1f}]** are not."),

        md(f"## The one look-ahead in the study, priced\n\n"
           f"Nothing in the *return* series peeks: the ranking formed at the last close of year "
           f"*y* first earns on the second trading day of *y+1*. But the fund the `cheapest` arm "
           f"holds is picked from the fee sheet published **today**, and FXAIX has charged 1.5 bp "
           f"only since 2019 (SWPPX 2 bp since 2017, IVV 3 bp since 2016). The direction was "
           f"public in advance — SPY and QQQ have always been the dearest of their families — but "
           f"*which* low-fee rung ends up lowest is chosen ex post. So the "
           f"{R['lad_cheap_lead']:+.2f} bp/yr headline is, in substance, **FXAIX selected with "
           f"hindsight**.\n\n"
           f"The control removes the selection entirely: race every non-flagship fund against the "
           f"flagship, buy-and-hold, plus the equal-weight blend of all of them — a rule needing "
           f"no fee sheet, no ranking and no forecast (`strategy.per_fund_gap_vs_leader`)."),

        code(
            "H = " + repr(HINDSIGHT_FREE) + "\n"
            "print('Buy-and-hold vs the family flagship — NO fee sheet, no ranking, no forecast')\n"
            "print('gap'.rjust(24) + 'bp/yr'.rjust(10) + 't (annual)'.rjust(12)\n"
            "      + 'yrs +'.rjust(8))\n"
            "for name, bp, t, pos in H:\n"
            "    print(name.rjust(24) + format(bp, '+.2f').rjust(10)\n"
            "          + format(t, '+.2f').rjust(12) + pos.rjust(8))\n"
            "print()\n"
            "print('1. EVERY fund beats its flagship: the SIGN owes nothing to hindsight.')\n"
            "print('2. The hindsight-free magnitude is +5.21 bp/yr (t = +2.68) — about HALF the')\n"
            "print('   +10.64 the fee-sheet rule reports. Half the headline is the fund pick.')\n"
            "print('3. The two ETFs already cheap in 2012 (IVV, VOO) do not clear t = 1.')\n"
            "print('The Nasdaq pair has no selection to make, and is the cleaner reading.')\n"
        ),

        md("## Sharpe is the wrong instrument here"),

        code(
            "S = " + repr(SHARPE) + "\n"
            "print('Excess-of-cash Sharpe vs BIL (both arms excess), S&P 500 ETF trio:')\n"
            "for name, v in S:\n"
            "    print('  ' + name.rjust(9) + '  ' + format(v, '.4f'))\n"
            "vals = [v for _, v in S]\n"
            "print('  spread: ' + format(max(vals) - min(vals), '.4f'))\n"
            "print()\n"
            "print('Every arm holds the same index, so they share ~99.99% of their variance.')\n"
            "print('A genuine 3-11 bp/yr edge lives in the 4th decimal of the Sharpe ratio and is')\n"
            "print('unmeasurable there. A tracking-difference edge is a return GAP, not a Sharpe.')\n"
        ),

        md("## Era cut"),

        code(
            "E = " + repr(ERAS) + "\n"
            "for name, e_lab, e_gap, e_t, l_lab, l_gap, l_t in E:\n"
            "    print(name + ':  ' + e_lab + ' ' + format(e_gap, '+.2f') + ' bp/yr (t='\n"
            "          + format(e_t, '+.2f') + ')   ' + l_lab + ' ' + format(l_gap, '+.2f')\n"
            "          + ' bp/yr (t=' + format(l_t, '+.2f') + ')')\n"
            "print()\n"
            "print('The ladder gap is era-robust (+10.94 then +10.39).')\n"
            "print('The trio gap only appears late: SPY has not cut its fee while its rivals have.')\n"
        ),

        md(f"## Cost sweep and the tax assumption\n\n"
           f"`cheapest` never switches fund, so its gap is cost-invariant by construction; "
           f"`winner` changed fund 10 times in 14 live years, so it pays. One disclosure: where "
           f"`cheapest` is a **tie** (the trio's IVV/VOO at 3 bp) it is held at a constant 50/50 "
           f"target — an implicit daily rebalance that is charged nothing. Between funds this "
           f"similar the freebie measures **{R['tie_rebal_freebie']:+.2f} bp/yr**, so it changes "
           f"no conclusion, but only the single-fund arms are literally buy-and-hold."),

        code(
            "SW = " + repr(COST_GRID) + "\n"
            "print('winner - cheapest, S&P 500 ETF trio, by one-way switching cost')\n"
            "print('cost (bp)'.rjust(10) + 'gap bp/yr'.rjust(12) + 't (annual)'.rjust(12))\n"
            "for c, gap, t in SW:\n"
            "    print(format(c, '.0f').rjust(10) + format(gap, '+.2f').rjust(12)\n"
            "          + format(t, '+.2f').rjust(12))\n"
            "print()\n"
            "print('A coin flip gross; reliably negative at any cost a real switch incurs.')\n"
        ),

        code(
            SETUP +
            "from td_persist import strategy as st\n"
            "gap = 9.45 - 3.00   # published SPY minus VOO expense ratio — an ASSUMPTION\n"
            "print('Years for the ' + format(gap, '.2f')\n"
            "      + ' bp/yr fee gap to repay a realised-gain tax bill')\n"
            "print('(ASSUMPTION grid: neither the embedded gain nor the rate is on the tape)')\n"
            "print(st.tax_breakeven_years(gap).round(0).to_string())\n"
        ),

        md("## Synthetic control — the machinery, and only the machinery\n\n"
           "**Synthetic data, not the real tape.** A panel of four funds tracking one index, "
           "separated by a planted fee ladder against an 8 bp annual noise floor. At "
           "`signal_strength=1` the ladder is 30 bp and the pipeline must find it; at "
           "`signal_strength=0` every fund charges the identical fee and the pipeline must stay "
           "silent. Eight seeds each — never one lucky draw."),

        code(
            SETUP +
            "import pandas as pd\n"
            "from td_persist import data, strategy as st\n"
            "out = []\n"
            "for ss in (1.0, 0.0):\n"
            "    rows = [st.synthetic_detect(*data.synthetic_panel(signal_strength=ss, seed=913 + s))\n"
            "            for s in range(8)]\n"
            "    f = pd.DataFrame(rows)\n"
            "    out.append({'world': 'planted 30 bp ladder' if ss else 'flat-fee null',\n"
            "                'rho': f['mean_spearman'].mean(),\n"
            "                'rho_min': f['mean_spearman'].min(),\n"
            "                'rho_max': f['mean_spearman'].max(),\n"
            "                'cheapest-leader bp': f['cheapest_minus_leader_bp'].mean(),\n"
            "                't': f['t_cheapest_minus_leader'].mean()})\n"
            "print('SYNTHETIC control (8 seeds each) — never supports a real-tape stamp')\n"
            "print(pd.DataFrame(out).set_index('world').round(3).to_string())\n"
        ),

        md("## Threats to validity, named\n\n"
           "- **Survivorship.** Eight vehicles, all of which still exist today. Trackers that "
           "tracked badly enough to close or merge are absent by construction, so measured TD "
           "dispersion is understated and the cheapest-fund result sits on the friendliest "
           "possible sample. Named on the Signal axis.\n"
           "- **Expense ratios are an ASSUMPTION that carries hindsight**, taken from issuer "
           "disclosure at build time; several were cut inside the sample, so today's ladder is a "
           "level rather than a history, and the `cheapest` arm's fund choice is partly ex post. "
           "The cost sweep bounds how wrong the assumed spread can be, and the hindsight-free "
           "control above prices the selection: the gap halves (+10.64 → +5.21 bp/yr) but keeps "
           "its sign and its *t* (+2.68).\n"
           "- **Multi-fund arms are held at target weight**, i.e. implicitly rebalanced daily at "
           "no cost. On the real trio's tied `cheapest` (IVV/VOO) that freebie is +0.06 bp/yr — "
           "immaterial here, but only single-fund arms are literally buy-and-hold.\n"
           "- **NAV-priced legs.** VFIAX, FXAIX and SWPPX carry no spread and no "
           "premium/discount, which is exactly why they widen the ladder cleanly — but they "
           "cannot be switched intraday, and the tax grid applies to them identically.\n"
           "- **SPLG is missing, not omitted.** The source serves a single stale bar for it; it "
           "is declared in `data.UNAVAILABLE` rather than silently substituted.\n"
           "- **Five complete years** is all QQQ/QQQM offers. A 4/4 sign run is suggestive and "
           "fully consistent with the 5 bp fee gap — it is not a large-sample result."),

        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The persistence claim fails where it matters: the S&P 500 ETF "
           f"trio shows rank persistence of **{R['trio_rho']:+.3f}** (*t* = "
           f"{R['trio_rho_t']:+.2f}, permutation *p* = {R['trio_perm_p']:.2f}), because a "
           f"{R['trio_fee_spread']:.2f} bp fee spread is unresolvable through a "
           f"{R['trio_td_sd']:.1f} bp measurement floor. Across a wider ladder the *fee* effect "
           f"is unambiguous — cheapest − leader **{R['lad_cheap_lead']:+.2f} bp/yr, *t* = "
           f"{R['lad_cheap_lead_t']:+.2f}** (HAC {R['lad_cheap_lead_hac']:+.2f}), "
           f"{R['lad_cheap_lead_pos']} years, era-robust "
           f"({R['lad_era_e_gap']:+.2f} then {R['lad_era_l_gap']:+.2f}), CI "
           f"[{R['lad_cheap_lead_ci'][0]:+.1f}, {R['lad_cheap_lead_ci'][1]:+.1f}], halving to "
           f"**{R['lad_ew_gap']:+.2f} (*t* = {R['lad_ew_t']:+.2f})** once the ex-post fee-sheet "
           f"pick is removed — and QQQM beat QQQ by **{R['ndx_raw_gap']:+.2f} bp/yr (*t* = "
           f"{R['ndx_raw_t']:+.2f})** in {R['ndx_raw_pos']} years, "
           f"{R['ndx_cheap_lead']:+.2f} (*t* = {R['ndx_cheap_lead_t']:+.2f}) over the "
           f"{R['ndx_cheap_lead_pos']} a rule could trade. But the permutation test identifies that as a "
           f"time-invariant level rather than a memory: the all-pairs rank correlation "
           f"({R['lad_rho_allpairs']:+.3f}) matches the consecutive-pair one "
           f"({R['lad_rho']:+.3f}). Half the claim is right, for a reason that makes the other "
           f"half redundant. Survivorship flatters the sample, and half the fee-gap magnitude is "
           f"fee-sheet hindsight; the synthetic control fires on a "
           f"planted ladder ({R['syn_pl_rho']:+.3f}, gap {R['syn_pl_gap']:+.1f} bp) and stays "
           f"silent on the null ({R['syn_nl_rho']:+.3f}, gap {R['syn_nl_gap']:+.2f} bp).\n"
           f"- **Tradability — Fragile.** The bankable form is a purchase decision, not a "
           f"strategy: buy a cheap share class, never trade it, collect "
           f"{R['trio_cheap_lead']:.1f}–{R['ndx_raw_gap']:.1f} bp/yr. The rotation form pays "
           f"{R['trio_win_cheap']:+.2f} bp/yr (*t* = {R['trio_win_cheap_t']:+.2f}) for "
           f"{R['trio_switches']} trades in {R['trio_live_years']} years, is negative beyond 1 bp "
           f"of switching cost, and — at a +50% embedded gain and a 20% rate — takes **155 "
           f"years** to repay the tax on moving an existing position."),
    ]
    nb = new_notebook()
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
