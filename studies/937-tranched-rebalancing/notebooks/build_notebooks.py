"""Generate the two narrative notebooks for Study 937 (Tranches).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
offline **synthetic** control, and they are labelled as such — no synthetic figure ever
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


# Frozen real-tape headline — mirror of docs/results.md. SPY/IEF monthly 200-day trend
# sleeve, 21 rebalance offsets, excess-of-cash (^IRX proxy), 5 bps one-way,
# books live 2003-06-16 -> 2026-06-30.
R = dict(
    tape_start="2002-07-30", start="2003-06-16", end="2026-06-30",
    n_tape=6018, n_days=5797, fp="e8f9552be0ac", in_risky=81.0,
    # the cone, by number of tranches
    n1_sharpe=0.631, n1_sd=0.066, n1_min=0.499, n1_max=0.755, n1_range=0.256,
    n1_cagr_sd=0.86, n1_tw=95.1, n1_turn=2.741,
    n4_sharpe=0.670, n4_sd=0.023, n4_range=0.092, n4_tw=27.3, n4_turn=2.71,
    n12_sharpe=0.675, n12_sd=0.010, n12_range=0.034, n12_tw=7.7, n12_turn=2.70,
    n21_sharpe=0.676, n21_sd=0.000, n21_range=0.000, n21_tw=0.0, n21_turn=2.700,
    shrink4=0.340, shrink12=0.152, shrink21=0.000,
    # what tranching buys
    gain=0.044, cagr_1=9.63, cagr_21=9.70, cagr_gain=0.07,
    vol_1=13.30, vol_21=12.32, dd_1=-31.0, dd_1_worst=-35.4, dd_21=-26.2,
    turn_change=-1.5,
    # bootstrap
    sd_ci_lo=0.054, sd_ci_hi=0.145, gain_ci_lo=0.014, gain_ci_hi=0.073, gain_neg=0.3,
    # forecastability
    best_off=2, worst_off=17, best_sh=0.755, worst_sh=0.499, bw_gap=0.256, bw_t=1.98,
    rho_eras=0.309, split="2014-12-17", early_winner=2, late_rank=2,
    late_sh_winner=0.620, late_sh_mean=0.562, late_sh_best=0.635, late_winner=9,
    rho_rules=0.556,
    # same-window honesty: the chase edge is measured on the second half, so the
    # tranching gain it is compared with must be too (never the full-sample +0.044).
    chase_late=0.058, gain_late=0.042, gain_early=0.044,
    rnd_sharpe=0.515, rnd_sd=0.103, rnd_range=0.462, rnd_tw=419.0, rnd_tranched=0.590,
    # the edge-free control as a distribution over 20 seeds, not one lucky schedule
    rnd_seeds=20, rnd_sd_mean=0.093, rnd_sd_med=0.094, rnd_sd_min=0.065, rnd_sd_max=0.121,
    rnd_n_wider=19,
    # one MORE day of execution delay
    lag_sh=0.622, lag_sd=0.071, lag_range=0.270, lag_full=0.666, lag_gain=0.044,
    # every 4-tranche rotation beat the average single date
    n4_min=0.634, n4_max=0.726,
    # eras
    era_e_n=2656, era_e_sh=0.717, era_e_sd=0.096, era_e_range=0.341,
    era_e_full=0.762, era_e_gain=0.045,
    era_l_n=3119, era_l_sh=0.604, era_l_sd=0.056, era_l_range=0.218,
    era_l_full=0.645, era_l_gain=0.041,
    # costs & tickets
    cost0_gain=0.045, cost0_sd=0.065, cost25_gain=0.042, cost25_sd=0.071,
    tick1=2.6, tick4=11.0, tick12=32.3, tick21=57.6,
    drag21_half=0.288, drag21_two=1.151, drag4_half=0.055,
    # cross-checks
    mom_sd=0.039, mom_range=0.165, mom_tw=77.7, mom_gain=0.013,
    mom_turn_1=2.81, mom_turn_21=2.82,
    bil_n=4581, bil_sd=0.051, bil_range=0.206, bil_gain=0.046, bil_proxy_diff=0.008,
    # synthetic control (6 seeds, 40 years)
    syn_pl_edge=0.450, syn_pl_min=0.249, syn_pl_sd1=0.048, syn_pl_gain=0.024,
    syn_nl_edge=-0.003, syn_nl_max=0.175, syn_nl_sd1=0.053, syn_nl_gain=0.005,
)


HEADER = f"""# Study 937 — Tranches 🍰

**"Rebalanced monthly" hides a coin-flip: *which* day of the month?**

Take an ordinary sleeve — hold **SPY** for the next month if it closed above its **200-day**
average on the rebalance day, otherwise hold **IEF**. Nothing exotic. But "monthly" is not
one schedule, it is **21** of them: the book rebalanced on the 3rd trading day of the cycle
is a different book from the one rebalanced on the 10th, running the *identical rule*.

Study [836](../../836-timing-luck/) showed on synthetic data that this choice alone moves
the reported Sharpe. This study asks the follow-up on the **real tape**: how big is the
gap, and does the textbook fix — **tranching**, splitting the book into N sleeves
rebalanced on N staggered dates — actually close it, and at what price?

Real tape: SPY x IEF, {R['tape_start']} → {R['end']}, books live from {R['start']}
({R['n_days']:,} days), excess-of-cash both sides, 5 bps one-way. Lag: the state known at
the close of *t* is the position that earns the return of *t+1* (one shift).

*Every real number below is frozen from `docs/results.md` (Fingerprint `{R['fp']}`,
as-of 2026-06-30). The live cells at the end run the offline **synthetic** control and are
labelled as such.*
"""

SYN_SETUP = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "from tranching import data, strategy as st\n"
)


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The same rule, twenty-one answers\n\n"
           "Run the sleeve on each of the 21 possible rebalance days and you get 21 track "
           "records. They are not close together. Over 23 years the luckiest calendar "
           f"choice finished **{R['n1_tw']:.0f}% richer** than the unluckiest — same rule, "
           "same assets, same costs, same everything except the day of the month."),
        code(
            "R = dict(n1_min=%r, n1_max=%r, n1_range=%r, n1_tw=%r, n1_cagr_sd=%r)\n"
            "print('worst rebalance date : excess Sharpe %%+.3f' %% R['n1_min'])\n"
            "print('best  rebalance date : excess Sharpe %%+.3f' %% R['n1_max'])\n"
            "print('gap from luck alone  : %%.3f Sharpe, %%.1f%%%% of terminal wealth'\n"
            "      %% (R['n1_range'], R['n1_tw']))\n"
            "print('year-in-year-out spread in CAGR: %%.2f pp (standard deviation)'\n"
            "      %% R['n1_cagr_sd'])"
            % (R["n1_min"], R["n1_max"], R["n1_range"], R["n1_tw"], R["n1_cagr_sd"])
        ),
        md("## 2. It is luck, not skill — a rule with **no** edge has a wider gap\n\n"
           "If the spread came from some dates being genuinely better at timing markets, a "
           "rule with no timing ability would show no spread. It shows more: an "
           "exposure-matched **random-timing** rule on the same two funds disperses "
           f"**{R['rnd_range']:.3f}** of a Sharpe point across the 21 dates and "
           f"**{R['rnd_tw']:.0f}%** of terminal wealth. One random schedule is itself a "
           f"coin-flip, so we ran **{R['rnd_seeds']}** of them: the edge-free cone is wider "
           f"than the trend rule's in **{R['rnd_n_wider']} of {R['rnd_seeds']}** "
           f"(mean sd {R['rnd_sd_mean']:.3f} against {R['n1_sd']:.3f}). The cone is an "
           "accident of the sampling schedule, and every monthly strategy on earth has one."),
        md("## 3. The fix: stop betting the book on one date\n\n"
           "Split the capital into **N sleeves**, each rebalanced on a different day of the "
           "cycle, each left to compound on its own. Nobody has to guess the lucky date, "
           "because you own all of them. Here is the spread of outcomes as N grows:"),
        code(
            "rows = [(1, %r, %r, %r), (4, %r, %r, %r), (12, %r, %r, %r), (21, %r, %r, %r)]\n"
            "print('tranches   Sharpe spread (sd)   terminal-wealth spread   traded/yr')\n"
            "for n, sd, tw, turn in rows:\n"
            "    print('%%8d   %%17.3f   %%20.1f%%%%   %%9.2fx' %% (n, sd, tw, turn))\n"
            "print('\\nfour sleeves already remove two-thirds of the luck; twenty-one remove all of it,')\n"
            "print('and the book trades LESS, not more.')"
            % (R["n1_sd"], R["n1_tw"], R["n1_turn"],
               R["n4_sd"], R["n4_tw"], R["n4_turn"],
               R["n12_sd"], R["n12_tw"], R["n12_turn"],
               R["n21_sd"], R["n21_tw"], R["n21_turn"])
        ),
        md("> 🔬 **For the quants.** The collapse is faster than the naive 1/√N: "
           f"sd(N)/sd(1) = 1.000 / {R['shrink4']:.3f} / {R['shrink12']:.3f} / "
           f"{R['shrink21']:.3f} against 1.000 / 0.500 / 0.289 / 0.218. At N = 21 it is "
           "*exactly* zero — every rotation of a 21-tranche book is the same set of 21 "
           "offsets, so there is nothing left to be lucky about. It is arithmetic, not an "
           "estimate."),
        md("## 4. What it costs — and what it is worth\n\n"
           f"Turnover barely moves (**{R['n1_turn']:.2f}x → {R['n21_turn']:.2f}x** of NAV a "
           f"year, {R['turn_change']:+.1f}%), because each sleeve only trades 1/N of the "
           "book. The risk-adjusted return improves a little. But be honest about *where* "
           "that improvement comes from:"),
        code(
            "print('one date (average of 21) -> 21 tranches')\n"
            "print('  excess Sharpe  %%+.3f  ->  %%+.3f   (%%+.3f)' %% (%r, %r, %r))\n"
            "print('  CAGR           %%5.2f%%%%  ->  %%5.2f%%%%   (%%+.2f pp/yr)' %% (%r, %r, %r))\n"
            "print('  volatility     %%5.2f%%%%  ->  %%5.2f%%%%' %% (%r, %r))\n"
            "print('  worst drawdown %%5.1f%%%%  ->  %%5.1f%%%%' %% (%r, %r))\n"
            "print()\n"
            "print('Almost all of the Sharpe gain is LOWER RISK, not higher return:')\n"
            "print('seven basis points a year of extra growth, one full point less volatility.')"
            % (R["n1_sharpe"], R["n21_sharpe"], R["gain"],
               R["cagr_1"], R["cagr_21"], R["cagr_gain"],
               R["vol_1"], R["vol_21"], R["dd_1_worst"], R["dd_21"])
        ),
        md("## 5. The bill nobody prints: broker tickets\n\n"
           "Percentage costs do not care how many sleeves you run — each trades 1/N of the "
           "book. **Tickets do.** A switch in 21 sleeves is 21 pairs of orders instead of "
           f"one: **{R['tick1']:.1f} → {R['tick21']:.1f} tickets a year**. A ticket costs a "
           "fixed sum, so what it costs *you* depends entirely on how big your account is "
           f"— at 0.5 bp of NAV per ticket the full 21-sleeve book pays **{R['drag21_half']:.2f}%/yr**, "
           f"which is four times the {R['cagr_gain']:.2f} pp/yr of growth it added. Inside a "
           "large fund it costs nothing. This is an **assumption**, not something we can "
           "read off the price tape — which is exactly why the sensible retail answer is "
           f"**four sleeves** ({R['tick4']:.0f} tickets/yr, {R['drag4_half']:.3f}%/yr) and "
           "not twenty-one."),
        md("## 6. Could you just pick the lucky date instead?\n\n"
           f"Maybe once. Split the history in half: the two halves' rankings of the 21 dates "
           f"correlate **ρ = {R['rho_eras']:+.3f}**. The first half's winner did rank "
           f"{R['late_rank']}/21 in the second half — worth **+{R['chase_late']:.3f}** over the "
           f"average date, which on those same rows was actually *more* than the "
           f"**+{R['gain_late']:.3f}** tranching gave over the same stretch. We say so plainly "
           "rather than compare it against the flattering full-sample number. But that is "
           "**one draw**, on a date chosen by looking at the first half, and the second half's "
           f"actual winner was a different date (offset {R['late_winner']}) — so the chaser is "
           "right back in the lottery, betting the whole book on a day of the month. Two "
           f"different rules do tend to like the same dates (ρ = {R['rho_rules']:+.3f}), which "
           "hints at a month-end flow footprint underneath; the fix needs no forecast at all."),
        md("## 7. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "On a synthetic world with a **planted** regime the sleeve genuinely beats a "
           "matched random control; on a **null** world it does not. In *both* worlds the "
           "timing-luck cone is there and tranching kills it — which is the point: the fix "
           "is schedule arithmetic, it neither needs edge nor creates any."),
        code(
            SYN_SETUP +
            "import numpy as np\n"
            "for tag, ss in [('planted regime', 1.0), ('flat null     ', 0.0)]:\n"
            "    rows = [st.synthetic_detect(p) for p, _ in\n"
            "            data.synthetic_panel(3, signal_strength=ss, n_years=40)]\n"
            "    edge = np.array([d['sleeve_minus_random'] for d in rows])\n"
            "    sd1 = np.mean([d['sharpe_sd_n1'] for d in rows])\n"
            "    sdf = np.mean([d['sharpe_sd_full'] for d in rows])\n"
            "    print('%s (3 worlds): sleeve vs random control %+.3f  |  cone sd %.3f -> %.3f'\n"
            "          % (tag, edge.mean(), sd1, sdf))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The cone is a fact of the real tape, not a simulation "
           f"artefact: **{R['n1_range']:.3f}** of a Sharpe point and **{R['n1_tw']:.0f}%** of "
           f"terminal wealth decided by the calendar, sd **{R['n1_sd']:.3f}**, and the same "
           f"picture in both eras, on a second rule, on a second cash leg, with an extra day "
           f"of execution delay, and wider on a rule with no edge at all. (It is a "
           f"*measurement*, not a bet won: a spread has no null to reject, and a standard "
           f"deviation cannot straddle zero, so we lean on size and replication rather than "
           f"on a p-value.) Tranching closes it exactly, for **{R['turn_change']:+.1f}%** "
           f"turnover and **{R['gain']:+.3f}** of Sharpe.\n"
           f"- **Tradability — Fragile.** What you bank is **{R['cagr_gain']:.2f} pp/yr** and a "
           f"quieter ride, not alpha — and the ticket bill (an assumption, not tape) can eat "
           f"it whole on a small account. Tranch so your track record stops being an accident "
           f"of the calendar; four sleeves buy most of that, twenty-one buy the rest at a "
           f"price only a large book can ignore."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 937 — Tranches — the teardown\n\n"
           "The timing-luck cone by tranche count, the joint block-bootstrap CIs, the "
           "(selection-biased) best-minus-worst *t*, the persistence test, the era cut, the "
           "cost and ticket sweeps, two cross-checks and the live synthetic control.\n\n"
           "**Design.** One rule (monthly 200-day trend, SPY vs IEF), 21 rebalance offsets, "
           "the state known at the close of *t* earning the return of *t+1* (one shift — so "
           "execution sits at the *t* close; a second day of delay is re-run in results.md "
           "and moves nothing), cost = one-way bps x traded notional "
           "(2 x |Δpos| on a switch), no shorts so no borrow, Sharpes excess-of-cash "
           "(^IRX accrual PROXY; BIL cross-check). An N-tranche book holds N sleeves on "
           "offsets floor(k·21/N), each self-financing; the book return is their NAV-share "
           "weighted average, evaluated at all 21 rotations of the anchor.\n\n"
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`, as-of "
           "2026-06-30); the live cells are synthetic and labelled." % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The cone by tranche count\n\n"
           "Dispersion is measured across the 21 rotations of each N-tranche book, so every "
           "row answers the same question: *how much does the arbitrary anchor date matter?*"),
        code(
            "print(f\"n = {R['n_days']:,} days, {R['start']} -> {R['end']}, in-risky {R['in_risky']:.1f}%\")\n"
            "print()\n"
            "print(' N   Sharpe    sd      min     max    range   CAGRsd   TWspread  traded/yr')\n"
            "rows = [(1, R['n1_sharpe'], R['n1_sd'], R['n1_min'], R['n1_max'], R['n1_range'], R['n1_cagr_sd'], R['n1_tw'], R['n1_turn']),\n"
            "        (4, R['n4_sharpe'], R['n4_sd'], 0.634, 0.726, R['n4_range'], 0.29, R['n4_tw'], R['n4_turn']),\n"
            "        (12, R['n12_sharpe'], R['n12_sd'], 0.662, 0.696, R['n12_range'], 0.10, R['n12_tw'], R['n12_turn']),\n"
            "        (21, R['n21_sharpe'], R['n21_sd'], 0.676, 0.676, R['n21_range'], 0.00, R['n21_tw'], R['n21_turn'])]\n"
            "for n, sh, sd, lo, hi, rg, cs, tw, tn in rows:\n"
            "    print(f'{n:2d}   {sh:+.3f}   {sd:.3f}  {lo:+.3f}  {hi:+.3f}   {rg:.3f}   {cs:4.2f}pp  {tw:6.1f}%   {tn:.3f}x')\n"
            "print()\n"
            "print(f\"shrink sd(N)/sd(1): 1.000 / {R['shrink4']:.3f} / {R['shrink12']:.3f} / {R['shrink21']:.3f}\")\n"
            "print(' 1/sqrt(N) reference: 1.000 / 0.500 / 0.289 / 0.218  -> the collapse beats it')"
        ),
        md("> 💡 **In plain words.** Averaging 21 highly-correlated books cancels the part "
           "of each that is pure sampling phase. It beats 1/√N because the residuals are not "
           "independent draws — they are the *same* signal sampled at different phases, and "
           "the full set of phases reconstructs the underlying rule exactly."),
        md("## 2. What the gain is made of\n\n"
           "The 21-tranche book **is** the NAV-weighted average of the 21 single-date books, "
           "so its mean return is theirs by construction. Any Sharpe gain must therefore come "
           "from the denominator — and it does."),
        code(
            "print(f\"Sharpe {R['n1_sharpe']:+.3f} -> {R['n21_sharpe']:+.3f}   gain {R['gain']:+.3f}\")\n"
            "print(f\"CAGR   {R['cagr_1']:.2f}% -> {R['cagr_21']:.2f}%   ({R['cagr_gain']:+.2f} pp/yr)\")\n"
            "print(f\"vol    {R['vol_1']:.2f}% -> {R['vol_21']:.2f}%   ({R['vol_21']-R['vol_1']:+.2f} pp)\")\n"
            "print(f\"maxDD  mean {R['dd_1']:.1f}% (worst placement {R['dd_1_worst']:.1f}%) -> {R['dd_21']:.1f}%\")\n"
            "print()\n"
            "m = R['n1_sharpe'] * R['vol_1']\n"
            "print(f\"implied mean excess return, one date: {m:.2f}%/yr\")\n"
            "print(f\"hold that numerator fixed, apply the tranched vol: {m/R['vol_21']:+.3f} \"\n"
            "      f\"vs the actual {R['n21_sharpe']:+.3f}\")\n"
            "print('-> the gain is the denominator. The numerator cannot move: the tranched book')\n"
            "print('   IS the NAV-weighted average of the 21 single-date books.')"
        ),
        md("## 3. Joint block bootstrap (1,000 draws, 21-day blocks)\n\n"
           "Rows are resampled **jointly** across the 21 books plus the tranched book, so the "
           "cross-sectional dependence that makes the cone narrow-ish survives the resample.\n\n"
           "**One of these two rows is inference and the other is decoration.** A standard "
           "deviation is non-negative, so an interval on the *cone sd* excludes zero for any "
           "21 non-identical books — it tests nothing. It is also badly right-skewed, the "
           "point sitting on its lower edge, because resampling blocks of days scrambles the "
           "very calendar phases that generate the cone. The **gain** is the row that could "
           "have printed negative and did not."),
        code(
            "print(f\"cone sd at N=1 : {R['n1_sd']:.3f}  95% band [{R['sd_ci_lo']:.3f}, {R['sd_ci_hi']:.3f}]\")\n"
            "print('                 ^ a SPREAD BAND, not a test: an sd cannot be negative.')\n"
            "print(f\"tranching gain : {R['gain']:+.3f}  95% CI [{R['gain_ci_lo']:+.3f}, {R['gain_ci_hi']:+.3f}]  \"\n"
            "      f\"share<0 {R['gain_neg']:.1f}%\")\n"
            "from statistics import NormalDist\n"
            "print(f\"                 ^ Gaussian equivalent |t| = {NormalDist().inv_cdf(1 - R['gain_neg']/100):.2f} \"\n"
            "      f\"(NOT 2.9 — that was a rounding of the wrong tail), on a quantity that is\")\n"
            "print('                   near-mechanical anyway: averaging imperfectly correlated')\n"
            "print('                   books at a fixed mean return almost has to cut the vol.')"
        ),
        md("## 4. Is the lucky offset forecastable? (it must not be)\n\n"
           "The best-minus-worst *t* is reported **only** as an upper bound: it compares the "
           "argmax with the argmin of 21 correlated books, so it is selection-biased by "
           "construction and cannot be read as evidence of a tradable spread."),
        code(
            "print(f\"ex-post best offset {R['best_off']} ({R['best_sh']:+.3f}) vs worst {R['worst_off']} ({R['worst_sh']:+.3f})\")\n"
            "print(f\"  gap {R['bw_gap']:.3f}, HAC t on the daily return difference {R['bw_t']:+.2f}  <- SELECTION-BIASED\")\n"
            "print(f\"persistence (split {R['split']}): rank correlation across halves rho = {R['rho_eras']:+.3f}\")\n"
            "print(f\"  first-half winner (offset {R['early_winner']}) ranked {R['late_rank']}/21 later: \"\n"
            "      f\"{R['late_sh_winner']:+.3f} vs the 21-date mean {R['late_sh_mean']:+.3f} \"\n"
            "      f\"(+{R['chase_late']:.3f}), while the true late winner was offset {R['late_winner']}\")\n"
            "print(f\"  SAME-WINDOW: chasing it earned {R['chase_late']:+.3f} in the second half vs \"\n"
            "      f\"{R['gain_late']:+.3f} for tranching over the SAME rows -> the chase WINS by \"\n"
            "      f\"{R['chase_late']-R['gain_late']:+.3f}.\")\n"
            "print('  It is one ex-post-selected draw, not an edge: the late winner was a different')\n"
            "print('  offset, rho is only +0.309, and the chaser re-enters the lottery he just won.')\n"
            "print('  (Comparing it against the FULL-SAMPLE +0.044 would flatter us. It does not.)')\n"
            "print(f\"cross-rule offset-ranking correlation: rho = {R['rho_rules']:+.3f} \"\n"
            "      f\"-> some shared month-end footprint, smaller than the cone it sits in\")\n"
            "print()\n"
            "print(f\"edge-free control (seed 937): exposure-matched RANDOM timing dispersal sd {R['rnd_sd']:.3f}, \"\n"
            "      f\"range {R['rnd_range']:.3f}, terminal spread {R['rnd_tw']:.0f}% -> tranched {R['rnd_tranched']:+.3f}\")\n"
            "print(f\"  over {R['rnd_seeds']} seeds: sd mean {R['rnd_sd_mean']:.3f}, median {R['rnd_sd_med']:.3f}, \"\n"
            "      f\"min {R['rnd_sd_min']:.3f}, max {R['rnd_sd_max']:.3f}; wider than the trend rule in \"\n"
            "      f\"{R['rnd_n_wider']}/{R['rnd_seeds']} (the quoted 0.103 is an upper-half draw)\")\n"
            "print('A rule with zero timing ability has a WIDER cone than the trend rule: artefact, not skill.')"
        ),
        md("## 5. Era cut, cost sweep, ticket assumption\n\n"
           "Costs are charged one-way on traded notional; nothing is shorted, so no borrow "
           "leg exists. The **ticket** figures are a labelled ASSUMPTION — broker fees are "
           "not on the price tape — and they are the only cost that scales with N."),
        code(
            "print(f\"2003-2013 (n={R['era_e_n']}): one date {R['era_e_sh']:+.3f} (sd {R['era_e_sd']:.3f}, \"\n"
            "      f\"range {R['era_e_range']:.3f}) -> tranched {R['era_e_full']:+.3f}  gain {R['era_e_gain']:+.3f}\")\n"
            "print(f\"2014-2026 (n={R['era_l_n']}): one date {R['era_l_sh']:+.3f} (sd {R['era_l_sd']:.3f}, \"\n"
            "      f\"range {R['era_l_range']:.3f}) -> tranched {R['era_l_full']:+.3f}  gain {R['era_l_gain']:+.3f}\")\n"
            "print()\n"
            "print(f\"cost 0 bps : gain {R['cost0_gain']:+.3f} (cone sd {R['cost0_sd']:.3f})\")\n"
            "print(f\"cost 25 bps: gain {R['cost25_gain']:+.3f} (cone sd {R['cost25_sd']:.3f})  \"\n"
            "      f\"-> friction does not drive this; the cone even widens, unlucky dates trade more\")\n"
            "print()\n"
            "for n, tk in [(1, R['tick1']), (4, R['tick4']), (12, R['tick12']), (21, R['tick21'])]:\n"
            "    print(f\"  N={n:2d}: {tk:5.1f} tickets/yr -> {tk*0.5/1e4*100:.3f}%/yr at 0.5bp/ticket, \"\n"
            "          f\"{tk*2.0/1e4*100:.3f}%/yr at 2bp/ticket\")\n"
            "print(f\"the 21-tranche ticket bill at 0.5bp ({R['drag21_half']:.3f}%/yr) exceeds the \"\n"
            "      f\"{R['cagr_gain']:.2f}pp/yr of CAGR it adds — size-dependent, and not modelled in the Sharpes above\")"
        ),
        md("## 6. Cross-checks"),
        code(
            "print(f\"12-1 momentum sleeve : cone sd {R['mom_sd']:.3f}, range {R['mom_range']:.3f}, \"\n"
            "      f\"terminal spread {R['mom_tw']:.1f}% -> 0.000 tranched; gain {R['mom_gain']:+.3f}; \"\n"
            "      f\"turnover {R['mom_turn_1']:.2f}x -> {R['mom_turn_21']:.2f}x\")\n"
            "print(f\"tradable BIL cash leg: n={R['bil_n']}, cone sd {R['bil_sd']:.3f}, range {R['bil_range']:.3f}, \"\n"
            "      f\"gain {R['bil_gain']:+.3f}; the ^IRX PROXY moves the Sharpe by {R['bil_proxy_diff']:.3f}\")\n"
            "print(f\"one MORE day of delay: single-date {R['lag_sh']:+.3f} (sd {R['lag_sd']:.3f}, \"\n"
            "      f\"range {R['lag_range']:.3f}) -> tranched {R['lag_full']:+.3f}, gain {R['lag_gain']:+.3f} \"\n"
            "      f\"-> not an execution-timing artefact\")\n"
            "print(f\"N=4 rotations span {R['n4_min']:+.3f} to {R['n4_max']:+.3f}: every one of the 21 \"\n"
            "      f\"four-tranche placements beat the average single date ({R['n1_sharpe']:+.3f})\")"
        ),
        md("## 7. Live synthetic control — planted vs null (offline, deterministic)\n\n"
           "Positive control: with a planted regime the sleeve must beat an exposure-matched "
           "random control. Negative control: on the flat null it must not. In both worlds "
           "the cone must exist and tranching must collapse it — that is what makes the fix "
           "orthogonal to the edge."),
        code(
            SYN_SETUP +
            "import numpy as np\n"
            "for tag, ss in [('planted', 1.0), ('null   ', 0.0)]:\n"
            "    rows = [st.synthetic_detect(p) for p, _ in\n"
            "            data.synthetic_panel(3, signal_strength=ss, n_years=40)]\n"
            "    edge = np.array([d['sleeve_minus_random'] for d in rows])\n"
            "    print('%s (3 worlds): sleeve-vs-random %+.3f [%+.3f, %+.3f] | cone sd %.3f -> %.3f | gain %+.3f'\n"
            "          % (tag, edge.mean(), edge.min(), edge.max(),\n"
            "             np.mean([d['sharpe_sd_n1'] for d in rows]),\n"
            "             np.mean([d['sharpe_sd_full'] for d in rows]),\n"
            "             np.mean([d['gain'] for d in rows])))\n"
            "print()\n"
            "print('frozen 6-seed summary from docs/results.md: planted edge %+.3f (min %+.3f), '\n"
            "      'null edge %+.3f (max %+.3f)' % (R['syn_pl_edge'], R['syn_pl_min'], R['syn_nl_edge'], R['syn_nl_max']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The rebalance-date cone is measurable on the real tape at "
           f"sd **{R['n1_sd']:.3f}**, range **{R['n1_range']:.3f}** Sharpe and "
           f"**{R['n1_tw']:.0f}%** of terminal wealth; it survives an era cut "
           f"({R['era_e_sd']:.3f} / {R['era_l_sd']:.3f}), a second rule ({R['mom_sd']:.3f}), "
           f"the tradable cash leg ({R['bil_sd']:.3f}), an extra day of execution delay "
           f"({R['lag_sd']:.3f}), and is *wider* for an edge-free rule in "
           f"{R['rnd_n_wider']}/{R['rnd_seeds']} seeds (mean {R['rnd_sd_mean']:.3f}) — the "
           f"signature of an artefact, which is exactly the claim. The stamp rests on **size "
           f"and replication, not on a p-value**: a dispersion has no null to reject, and the "
           f"sd's bootstrap band cannot straddle zero whatever the data say. Tranching removes "
           f"it identically (sd 0.000 at N=21, {R['shrink4']:.3f} of it left at N=4) with a "
           f"Sharpe gain of **{R['gain']:+.3f}** (CI [{R['gain_ci_lo']:+.3f}, "
           f"{R['gain_ci_hi']:+.3f}], {R['gain_neg']:.1f}% of draws negative — the study's one "
           f"real interval), stable across eras and from 0 to 25 bps. Forecasting the lucky "
           f"date did beat the fix in the one out-of-sample half we have "
           f"({R['chase_late']:+.3f} against {R['gain_late']:+.3f} on the same rows, "
           f"ρ = {R['rho_eras']:+.3f}) — one ex-post-selected draw, reported as it fell, and "
           f"not something we would size a book on.\n"
           f"- **Tradability — Fragile.** Proportional cost is a non-issue "
           f"({R['n1_turn']:.3f}x → {R['n21_turn']:.3f}x NAV/yr), but the gain is variance, not "
           f"return (**{R['cagr_gain']:+.2f} pp/yr** of CAGR), and the unpriced ticket bill "
           f"({R['tick1']:.1f} → {R['tick21']:.1f} tickets/yr; {R['drag21_half']:.3f}%/yr at "
           f"0.5 bp each) can exceed it outright on a small account. Tax, likewise, is not "
           f"modelled and runs against high N. Four tranches is the defensible operating "
           f"point: {(1-R['shrink4'])*100:.0f}% of the cone gone for {R['tick4']:.0f} tickets a year."),
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
