"""Generate the two narrative notebooks for Study 960 (The Unstaked ETF).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
synthetic control, and they are never placed under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. ETHA / ETHE / ETH-USD / BIL,
# daily closes, ETF era 2024-07-23 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2024-07-23", end="2026-06-30", n_days=486, fp="4e34c697361e",
    # 1. fund vs coin
    etha_td=0.27, etha_geo=-0.31, etha_lo=-8.86, etha_hi=9.34,
    etha_mt=0.11, etha_hac=0.03, etha_iid=0.01, etha_res=9.10,
    ethe_td=-1.37, ethe_geo=-1.31, ethe_lo=-10.53, ethe_hi=7.77,
    ethe_mt=-0.56, ethe_hac=-0.16, ethe_res=9.15,
    coin_sd_bps=190, coin_ac1=-0.56,
    # 1b. the resolution limit is RULER-dependent — the whole sweep, not one width
    res_sweep=[("block 5", 18.0), ("block 10", 13.2), ("block 21", 9.1),
               ("block 42", 6.1), ("block 63", 5.4), ("monthly", 5.0)],
    res_min=5.0, res_max=18.0,
    # coverage: measured 2 of a ~9-line cohort, and they are the fee extremes
    cohort_n=9, cohort_missing="FETH, ETHW, ETHV, QETH, EZET, CETH",
    # 2. fund vs fund
    ff_td=-1.64, ff_geo=-1.00, ff_lo=-2.27, ff_hi=-0.97,
    ff_mt=-5.16, ff_iid=-1.32, ff_hac=-2.89, ff_sd_bps=11,
    ff_months_neg=22, ff_months_n=24, ff_jack_lo=-1.79, ff_jack_hi=-1.51,
    ff_block_hw=[1.15, 0.84, 0.65, 0.61, 0.59], ff_monthly_hw=0.62,
    ff_terminal_pct=3.1, ff_years=1.92,
    fee_spread=2.25, ff_share_of_fee=73,
    # 3. ledger (assumed staking 3.0%)
    fee_etha=0.25, fee_ethe=2.50, stake=3.00,
    etha_short=-2.73, etha_share=92, etha_unexpl=0.52,
    ethe_short=-4.37, ethe_share=55, ethe_unexpl=1.13,
    # 4. sweeps
    sweep=[(2.0, -1.73, 89), (2.5, -2.23, 91), (3.0, -2.73, 92),
           (3.5, -3.23, 93), (4.5, -4.23, 95)],
    fee_sweep=[(0.12, 96), (0.25, 92)],
    # 5. era cut
    era_e_ff=-2.43, era_e_ff_lo=-3.51, era_e_ff_hi=-1.39, era_e_ff_mt=-6.25, era_e_n=235,
    era_l_ff=-0.90, era_l_ff_lo=-1.72, era_l_ff_hi=-0.02, era_l_ff_mt=-2.02, era_l_n=250,
    era_e_etha=0.36, era_l_etha=0.19, era_e_ethe=-2.07, era_l_ethe=-0.71,
    cal=[(2024, -2.14, 112), (2025, -1.97, 250), (2026, -0.52, 123)],
    # 6. capture trade
    ls_gross=1.85, ls_hac=3.32, ls_iid=1.50, ls_vol=1.72,
    ls_mt=5.69, ls_months_pos=21, ls_months_n=24,
    borrow=[(0, 1.45), (50, 0.95), (100, 0.45), (200, -0.55), (400, -2.55), (800, -6.55)],
    borrow_be=145,
    # 7. excess-of-cash race
    sh_etha=-0.260, sh_ethe=-0.290, sh_coin=-0.299,
    adv_etha=0.038, adv_etha_t=0.24, fund_vol=72.5,
    # 8. synthetic control
    syn_planted=2.25, syn_ff=-2.24, syn_ff_lo=-3.05, syn_ff_hi=-1.45, syn_ff_mt=-5.84,
    syn_fc=0.28, syn_fc_lo=-12.01, syn_fc_hi=12.95, syn_ratio=16,
    syn_null_mean=-0.011, syn_null_sd=0.065, syn_null_fire=0,
)

HEADER = f"""# Study 960 — The Unstaked ETF ⛓️

**A US spot-Ethereum ETF owns coins but does not stake them. What does that cost you?**

An on-chain holder can post their ether to the consensus layer and earn a protocol reward
of roughly 3% a year. Over most of this sample a US spot-ETH fund did not, so its holder
gave that reward up on top of the sponsor fee. The obvious test: measure how far each fund
fell behind the coin.

We run it on **ETHA** (0.25%/yr) and **ETHE** (2.50%/yr) against **ETH-USD**, daily closes,
{R['start']} → {R['end']} ({R['n_days']} sessions). Then we find out that the obvious test
cannot work — and what does.

*Real numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`).
The live cells run the offline synthetic control only. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The straightforward measurement — and why it says nothing\n\n"
           "Take each fund's price, take the coin's price, and see how far apart they drift "
           "per year. That is a *tracking difference*, and it is how anyone would check "
           "whether a fund is quietly costing you 3% a year."),
        code(
            "R = dict(etha_td=%r, etha_lo=%r, etha_hi=%r, etha_mt=%r,\n"
            "         ethe_td=%r, ethe_lo=%r, ethe_hi=%r, ethe_mt=%r)\n"
            "for name in ('etha', 'ethe'):\n"
            "    print('%%-5s vs ETH-USD: %%+.2f%%%%/yr    95%%%% confidence [%%+.2f, %%+.2f]   monthly t = %%+.2f'\n"
            "          %% (name.upper(), R[name+'_td'], R[name+'_lo'], R[name+'_hi'], R[name+'_mt']))"
            % (R["etha_td"], R["etha_lo"], R["etha_hi"], R["etha_mt"],
               R["ethe_td"], R["ethe_lo"], R["ethe_hi"], R["ethe_mt"])
        ),
        md(f"Look at the width of those brackets. The answer is *somewhere between minus nine "
           f"and plus nine percent a year*. That is not a measurement, it is a shrug. The "
           f"cause is timing: ether trades every hour of every day, and the price your data "
           f"vendor stamps as the coin's daily close is taken several hours after the "
           f"**16:00 New York** bell that closes the fund. On an asset that moves ~{R['fund_vol']:.0f}% a "
           f"year, those few hours dwarf everything. The day-to-day gap swings by "
           f"**{R['coin_sd_bps']} basis points** and then mostly reverses ({R['coin_ac1']:+.2f} "
           f"autocorrelation) — noise, not information.\n\n"
           "> 🔬 **For the quants:** the resolution limit — half the confidence-interval width "
           f"— is **not one number**. It depends on the ruler: ±{R['res_max']:.0f}%/yr with "
           f"5-day bootstrap blocks, ±{R['etha_res']:.1f} at 21, ±{R['res_min']:.1f} on "
           "non-overlapping months. We publish the sweep rather than the tidiest member of it. "
           f"Any drag smaller than ±{R['res_min']:.1f}%/yr is invisible on this tape however you "
           f"hold the ruler — and the drag in question is ~{R['stake']:.0f}%/yr."),
        md("## 2. The deeper problem — the yardstick is unstaked too\n\n"
           "Suppose the noise vanished. The measurement would *still* miss the staking yield, "
           "and this is the part almost everyone gets wrong.\n\n"
           "The consensus reward is not paid into the price of a coin. It is paid **in extra "
           "coins**, to the validator who posted them. So `ETH-USD` — the market price of one "
           "ether — is the price of an **unstaked** coin. A fund that does not stake and a "
           "holder who does not stake are, on that yardstick, doing the same thing. The "
           "forgone reward cannot show up in the gap between them. It is not hiding in the "
           "noise; it is not in the data at all."),
        md(f"## 3. So where does the number come from? Two constants\n\n"
           f"Write the honest ledger. The tape gives one column; the other two are quoted "
           f"from the issuer's fee schedule and from the Ethereum network — declared "
           f"**assumptions**, not observations.\n\n"
           f"| | Measured vs the (unstaked) coin | Assumed fee | Assumed staking | Behind a self-staking holder |\n"
           f"|---|--:|--:|--:|--:|\n"
           f"| **ETHA** | {R['etha_td']:+.2f}%/yr | {R['fee_etha']:.2f}% | {R['stake']:.2f}% | **{R['etha_short']:+.2f}%/yr** |\n"
           f"| **ETHE** | {R['ethe_td']:+.2f}%/yr | {R['fee_ethe']:.2f}% | {R['stake']:.2f}% | **{R['ethe_short']:+.2f}%/yr** |\n\n"
           f"On the cheap wrapper the forgone staking is **{R['etha_share']}%** of the whole "
           f"declared cost of ownership — the fee is almost a rounding error next to it. On the "
           f"dear one it is **{R['ethe_share']}%**, because a 2.5% fee is finally big enough to "
           f"compete. Both figures are true arithmetic. Neither is a finding: change the assumed "
           f"yield and they change with it, and the tape does not vote."),
        md(f"## 4. What the same measurement *can* see\n\n"
           f"Now point the identical estimator at **ETHE against ETHA** — two wrappers on the "
           f"same coin, both stamped at the same 16:00 bell. The timing error cancels exactly. "
           f"The daily wobble collapses from {R['coin_sd_bps']} basis points to "
           f"**{R['ff_sd_bps']}**, and a real number appears:\n\n"
           f"- **{R['ff_td']:+.2f}%/yr**, 95% confidence **[{R['ff_lo']:+.2f}, {R['ff_hi']:+.2f}]**, "
           f"monthly *t* = **{R['ff_mt']:+.2f}**.\n\n"
           f"The two funds' published fees differ by **{R['fee_spread']:.2f}%/yr**, and that "
           f"figure sits inside the interval. The expensive wrapper really does cost about what "
           f"it says it costs — and you can prove it, because for once the benchmark shares the "
           f"fund's closing bell. It holds up when you poke it: "
           f"**{R['ff_months_neg']} of {R['ff_months_n']} months** are negative, and dropping any "
           f"single month moves the answer only between {R['ff_jack_lo']:+.2f} and "
           f"{R['ff_jack_hi']:+.2f}%/yr.\n\n"
           f"> ⚠️ **Two things we are not hiding.** First, these are **2 of about "
           f"{R['cohort_n']} funds** that launched that day, and they are the cheapest and the "
           f"dearest — the widest gap available ({R['cohort_missing']} are not measured here). "
           f"Second, the crudest statistical test of all, which treats every day as independent, "
           f"gives only *t* = {R['ff_iid']:+.2f}. We do not use it, and notebook 02 §3 says "
           f"exactly why."),
        md(f"## 5. Something changed halfway through\n\n"
           f"That spread is not constant. Cut the sample in July 2025 and it goes from "
           f"**{R['era_e_ff']:+.2f}%/yr** to **{R['era_l_ff']:+.2f}%/yr**; year by year it reads "
           + ", ".join(f"**{v:+.2f}%** ({y})" for y, v, _ in R["cal"]) + ".\n\n"
           "A fee cut, a waiver rolling off, or one fund starting to stake its coins would all "
           "produce exactly this shape. **This tape cannot tell you which**, and pretending "
           "otherwise would be the very thing this study is complaining about."),
        md("## 6. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "*This cell runs on invented data, not the real tape.* We build one coin, a "
           "deliberately mis-timed vendor close, and two wrappers separated by a drag spread we "
           "choose ourselves. The fund-vs-fund estimator should find it; the fund-vs-coin "
           "estimator, fed the same world, should lose it completely."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from unstaked import data, strategy as st\n"
            "px, truth = data.synthetic_panel(signal_strength=1.0, seed=960)\n"
            "d = st.synthetic_detect(px)\n"
            "print('we planted a spread of %.2f%%/yr between the two wrappers' % truth['spread_planted_pct'])\n"
            "print('  fund vs fund : %+.2f%%/yr   95%% [%+.2f, %+.2f]  -> found it'\n"
            "      % (d['fund_vs_fund_pct'], d['fund_vs_fund_ci'][0], d['fund_vs_fund_ci'][1]))\n"
            "print('  fund vs coin : %+.2f%%/yr   95%% [%+.2f, %+.2f]  -> lost it (%.0fx wider)'\n"
            "      % (d['fund_vs_coin_pct'], d['fund_vs_coin_ci'][0], d['fund_vs_coin_ci'][1],\n"
            "         d['fund_vs_coin_halfwidth'] / d['fund_vs_fund_halfwidth']))\n"
            "null = st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=960)[0])\n"
            "print('  null (identical wrappers), fund vs fund: %+.3f%%/yr -> quiet, as it must be'\n"
            "      % null['fund_vs_fund_pct'])"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The staking gap the question asks about is **not visible** "
           f"({R['etha_td']:+.2f}%/yr, interval [{R['etha_lo']:+.1f}, {R['etha_hi']:+.1f}]) and "
           f"never could be, because the coin benchmark is unstaked. What *is* visible is real "
           f"and robust: the **{R['ff_td']:+.2f}%/yr** fee gap between the two wrappers, "
           f"*t* = {R['ff_mt']:+.2f}, matching {R['ff_share_of_fee']}% of the documented "
           f"{R['fee_spread']:.2f}%/yr difference.\n"
           f"- **Tradability — Fragile.** There is one honest action: **own the cheap wrapper**. "
           f"Worth the measured {abs(R['ff_td']):.2f}%/yr — {R['ff_terminal_pct']:.1f}% of your "
           f"money over {R['ff_years']:.2f} years — against a *documented* fee gap of "
           f"{R['fee_spread']:.2f}%/yr, for a single trade, no borrow, no cleverness. But it is "
           f"23 months of one asset and one fee-extreme pair, and it is already narrowing. "
           f"The ~3%/yr staking prize is not available from inside any of these funds; "
           f"collecting it means leaving the wrapper for self-custody or a staking provider, "
           f"with the lock-up, slashing and tax questions that brings.\n"
           f"- **The lesson worth keeping.** Before you quote anyone a tracking difference, "
           f"compute the confidence interval — then compute it a second way. Here it lands "
           f"anywhere from ±{R['res_min']:.0f} to ±{R['res_max']:.0f}%/yr depending on the "
           f"method, and *every* version is bigger than the ~{R['stake']:.0f}%/yr everyone was "
           f"arguing about. Quoting the single tidiest width would have been picking a ruler to "
           f"fit the story."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 960 — The Unstaked ETF — the teardown\n\n"
           "The tracking-difference estimator and its resolution limit, i.i.d. vs HAC vs block "
           "bootstrap on a negatively autocorrelated difference, the same-close benchmark that "
           "rescues the measurement, the wrapper-cost ledger and its assumption sweeps, the "
           "borrow sweep on the capture trade, the excess-of-cash race, and the live synthetic "
           "control. Every real number is frozen from `docs/results.md` (Fingerprint `%s`), "
           "ETF era %s → %s, %d sessions." % (R["fp"], R["start"], R["end"], R["n_days"])),
        code("R = %r" % (R,)),
        md("## 1. Estimator and sample\n\n"
           "Tracking difference is measured as the mean daily **log**-return difference × 252, "
           "so it telescopes to the endpoint wealth ratio. Both funds distribute nothing "
           "(`auto_adjust` close = price = total return); **ETH-USD is price-only and unstaked**; "
           "**BIL** is total return and is the cash leg.\n\n"
           "**Survivorship and coverage, separately.** *Survivorship:* both lines have traded "
           "continuously since day one and the fee ranking was published before launch, so "
           "nothing is selected on an outcome. *Coverage, which is the real caveat:* the "
           f"2024-07-23 cohort has about **{R['cohort_n']}** members and this study measures "
           f"**two** — the two in the desk cache ({R['cohort_missing']} are absent) — and they "
           "are the cohort's **cheapest and dearest**, i.e. its widest fee gap. Study 959 runs "
           "the ten-wrapper cross-section on bitcoin; there is no such replication here.\n\n"
           "> 💡 **In plain words:** we are asking how fast the fund falls behind the coin, per "
           "year, and how sure we are of that number."),
        code(
            "print(f\"ETHA vs ETH-USD : {R['etha_td']:+.2f}%/yr log (simple-ret diff {R['etha_geo']:+.2f}%)  \"\n"
            "      f\"95% CI [{R['etha_lo']:+.2f}, {R['etha_hi']:+.2f}]\")\n"
            "print(f\"   daily sd {R['coin_sd_bps']} bps  ac(1) {R['coin_ac1']:+.2f}  \"\n"
            "      f\"iid t {R['etha_iid']:+.2f}  HAC t {R['etha_hac']:+.2f}  monthly t {R['etha_mt']:+.2f}\")\n"
            "print(f\"ETHE vs ETH-USD : {R['ethe_td']:+.2f}%/yr log (simple-ret diff {R['ethe_geo']:+.2f}%)  \"\n"
            "      f\"95% CI [{R['ethe_lo']:+.2f}, {R['ethe_hi']:+.2f}]  monthly t {R['ethe_mt']:+.2f}\")\n"
            "print('\\nresolution limit at |t|=2 IS RULER-DEPENDENT -- the whole sweep:')\n"
            "for label, hw in R['res_sweep']:\n"
            "    print(f\"   {label:>9s}: +/-{hw:5.1f}%/yr   ({hw/R['stake']:.1f}x the \"\n"
            "          f\"{R['stake']:.1f}%/yr drag under discussion)\")\n"
            "print(f\"-> quote the RANGE +/-{R['res_min']:.1f} to +/-{R['res_max']:.1f}%/yr. \"\n"
            "      'Every ruler exceeds the effect, which is the only reason the claim stands.')"
        ),
        md("## 2. Why the interval is that wide — asynchronous closes\n\n"
           "The coin's daily bar is stamped hours after the 16:00 fund close, so each daily "
           "difference carries a transient error that reverses the next session (Scholes-Williams "
           "/ Dimson non-synchronicity, in its *late* rather than *stale* form). The error "
           "telescopes out of an endpoint comparison — which is why the two readings agree to well "
           "under a point — but it inflates every daily variance. The second column deserves its "
           "label: ETHA reads +0.27%/yr as a mean daily **log** difference and −0.31%/yr as a "
           "difference of annualised **simple** returns. That 0.6 pp swing is two things at once — "
           "the annualisation convention is level-dependent on an asset that halved, and the "
           "endpoint version is hostage to premium/discount noise on exactly two days. The log "
           "measure is the one whose sum is terminal wealth, so it is the one we quote.\n\n"
           "> 💡 **In plain words:** the two prices are photographed at different moments, so most "
           "of what looks like tracking error is just the clock."),
        md("## 3. The same-close benchmark — and which *t* we are quoting\n\n"
           "ETHE − ETHA: identical asset, identical closing bell, so the timing term is "
           "differenced away. The three variance treatments then disagree by a factor of four, "
           "and they are **not** equally defensible — nor is the one we quote the conservative "
           "one, so here is the direction stated plainly:\n\n"
           "- **i.i.d. daily *t* = −1.32** — the widest ruler, and it does *not* clear \\|*t*\\| = 2.\n"
           "- **HAC *t* = −2.89** — a *negative* autocovariance **shrinks** the HAC variance below "
           "i.i.d., so HAC is the **flattering** direction here, not a safety margin.\n"
           "- **Non-overlapping monthly *t* = −5.16** and the block bootstrap — flattering in the "
           "same direction, and what the verdict is read from.\n\n"
           "The case for discarding the i.i.d. reading is that the daily difference carries a "
           "**−0.47 one-session reversal** (premium/discount noise, essentially MA(1)): the "
           "variance of its *k*-day sum grows far more slowly than *k*, so the i.i.d. daily *t* "
           "assumes away a mechanical feature of the data and is over-wide rather than safe. "
           "Three checks decide whether that choice is doing the work — month signs, a "
           "month-jackknife, and a block sweep — and all three are printed below."),
        code(
            "print(f\"ETHE - ETHA: {R['ff_td']:+.2f}%/yr log (simple-return diff {R['ff_geo']:+.2f}%, \"\n"
            "      f\"same fact: {R['ff_terminal_pct']}% of terminal wealth over {R['ff_years']} yrs)\")\n"
            "print(f\"   95% CI [{R['ff_lo']:+.2f}, {R['ff_hi']:+.2f}]  \"\n"
            "      f\"daily sd {R['ff_sd_bps']} bps (vs {R['coin_sd_bps']} against the coin)\")\n"
            "print(f\"   iid t {R['ff_iid']:+.2f} (over-wide) | HAC t {R['ff_hac']:+.2f} (flattering) \"\n"
            "      f\"| monthly t {R['ff_mt']:+.2f} <- quoted\")\n"
            "print(f\"   robustness: {R['ff_months_neg']}/{R['ff_months_n']} months negative; \"\n"
            "      f\"month-jackknife {R['ff_jack_lo']:+.2f} to {R['ff_jack_hi']:+.2f}%/yr\")\n"
            "print('   block sweep of the half-width (b=5,10,21,42,63): '\n"
            "      + ', '.join(f'{h:.2f}' for h in R['ff_block_hw'])\n"
            "      + f\", monthly {R['ff_monthly_hw']:.2f} -> excludes zero on EVERY ruler\")\n"
            "print(f\"   documented fee spread -{R['fee_spread']:.2f}%/yr -> measured is \"\n"
            "      f\"{R['ff_share_of_fee']}% of it, and -{R['fee_spread']:.2f} lies inside the CI\")"
        ),
        md("## 4. The wrapper-cost ledger\n\n"
           "`wrapper_ledger` takes the MEASURED tracking difference and adds two declared "
           "ASSUMPTIONS: the sponsor fee and the net staking yield. The key identity is that "
           "ETH-USD is an **unstaked** price, so the staking term is *additive relative to a "
           "self-staking holder* and cannot appear in the measured column at all.\n\n"
           "> 💡 **In plain words:** the tape can price the fee. The staking cost has to be "
           "quoted from outside, and we say so every time we quote it."),
        code(
            "print(f\"{'fund':6s} {'measured TD':>12s} {'fee(A)':>8s} {'stake(A)':>9s} \"\n"
            "      f\"{'vs staked':>11s} {'stake share':>12s} {'unexplained':>12s}\")\n"
            "for tk, td, fee, sh, un in ((\"ETHA\", R['etha_td'], R['fee_etha'], R['etha_share'], R['etha_unexpl']),\n"
            "                            (\"ETHE\", R['ethe_td'], R['fee_ethe'], R['ethe_share'], R['ethe_unexpl'])):\n"
            "    tot = td - R['stake']\n"
            "    print(f\"{tk:6s} {td:+11.2f}% {fee:7.2f}% {R['stake']:8.2f}% {tot:+10.2f}% {sh:11d}% {un:+11.2f}%\")\n"
            "print('\\n(A) = ASSUMPTION, not tape. \"unexplained\" = measured TD minus the posted fee;')\n"
            "print(f\"both sit far inside even the NARROWEST resolution limit \"\n"
            "      f\"(+/-{R['res_min']:.1f}%/yr, monthly ruler), i.e. no evidence of anything.\")"
        ),
        md("## 5. Assumption sweeps\n\n"
           "The staking yield is the one number in this study that comes from outside the tape, "
           "so it never gets a point value without a sweep. The fee gets one too, because ETHA's "
           "early waiver makes its effective rate a range rather than a number."),
        code(
            "print('assumed staking -> ETHA shortfall vs a staking holder, and staking share of declared cost')\n"
            "for y, tot, share in R['sweep']:\n"
            "    print(f\"   {y:.1f}%/yr  ->  {tot:+.2f}%/yr   ({share}%)\")\n"
            "print('\\nassumed ETHA effective fee -> staking share')\n"
            "for f, share in R['fee_sweep']:\n"
            "    print(f\"   {f:.2f}%/yr  ->  {share}%\")\n"
            "print('\\nThe conclusion is insensitive to both, because the assumptions are the only inputs.')"
        ),
        md("## 6. Era cut and calendar table\n\n"
           "A fee is flat across eras; a forgone staking yield collapses when a fund starts "
           "staking. The fund-vs-coin rows stay uninformative in both halves (±11 to ±16 %/yr). "
           "The fund-vs-fund spread narrows by 1.5 pp — consistent with a fee change, a waiver "
           "expiring, or a staking policy change, and this tape cannot separate them."),
        code(
            "print(f\"ETHE-ETHA early (n={R['era_e_n']}): {R['era_e_ff']:+.2f}%/yr  \"\n"
            "      f\"CI [{R['era_e_ff_lo']:+.2f}, {R['era_e_ff_hi']:+.2f}]  monthly t {R['era_e_ff_mt']:+.2f}\")\n"
            "print(f\"ETHE-ETHA late  (n={R['era_l_n']}): {R['era_l_ff']:+.2f}%/yr  \"\n"
            "      f\"CI [{R['era_l_ff_lo']:+.2f}, {R['era_l_ff_hi']:+.2f}]  monthly t {R['era_l_ff_mt']:+.2f}\")\n"
            "print(f\"ETHA vs coin  early {R['era_e_etha']:+.2f}%/yr  late {R['era_l_etha']:+.2f}%/yr  (both noise)\")\n"
            "print(f\"ETHE vs coin  early {R['era_e_ethe']:+.2f}%/yr  late {R['era_l_ethe']:+.2f}%/yr  (both noise)\")\n"
            "print('\\ncalendar-year ETHE - ETHA:')\n"
            "for y, v, n in R['cal']:\n"
            "    print(f\"   {y}: {v:+.2f}%/yr   ({n} sessions)\")"
        ),
        md("## 7. Is the spread bankable? Long ETHA / short ETHE\n\n"
           "Dollar-neutral, one unit a side. Weights are formed from the two published fees "
           "through day *t* and held from *t+1* — **the study's single execution lag**. Costs are "
           "10 bps one-way × NAV per leg, in and out; the short leg pays an explicit annual "
           "borrow, swept, because a Grayscale-style trust is not general collateral. The book is "
           "dollar-neutral, so short proceeds fund the long leg and **no cash rebate is credited**. "
           "Gross reads slightly above the log tracking difference (+1.85 vs −1.64) because a "
           "daily-rebalanced arithmetic pair picks up the small variance difference between the "
           "two legs. Same ruler discipline as §3: the non-overlapping monthly *t* is quoted, HAC "
           "is flagged as the flattering direction, and the over-wide i.i.d. reading is shown too."),
        code(
            "print(f\"gross {R['ls_gross']:+.2f}%/yr  vol {R['ls_vol']:.2f}%\")\n"
            "print(f\"   monthly t {R['ls_mt']:+.2f} ({R['ls_months_pos']}/{R['ls_months_n']} months \"\n"
            "      f\"positive) <- quoted | HAC t {R['ls_hac']:+.2f} (flattering) | \"\n"
            "      f\"iid t {R['ls_iid']:+.2f} (over-wide, under 2)\")\n"
            "for b, net in R['borrow']:\n"
            "    flag = '  <- break-even is ~%d bps' % R['borrow_be'] if b == 200 else ''\n"
            "    print(f\"   borrow {b:4d} bps/yr -> net {net:+.2f}%/yr{flag}\")"
        ),
        md("## 8. Excess-of-cash race (both arms minus BIL)\n\n"
           "Included and read with a straight face: both arms hold the same coin, so the race can "
           "only surface the drag, and a 72%-vol asset over two years cannot. Every Sharpe is "
           "negative — the ETF era so far has been an ETH bear market. The fund leg is charged a "
           "5 bps one-way entry; the coin leg is charged nothing, which flatters the coin (the "
           "honest direction here).\n\n"
           "> 💡 **In plain words:** you cannot spot a 3%-a-year fee by comparing Sharpe ratios of "
           "two things that are the same thing."),
        code(
            "print(f\"ETHA  excess Sharpe {R['sh_etha']:+.3f}\")\n"
            "print(f\"ETHE  excess Sharpe {R['sh_ethe']:+.3f}\")\n"
            "print(f\"coin  excess Sharpe {R['sh_coin']:+.3f}\")\n"
            "print(f\"ETHA advantage vs coin {R['adv_etha']:+.4f}  HAC t {R['adv_etha_t']:+.2f}  -> noise\")"
        ),
        md("## 9. Live synthetic control — the machinery is unbiased\n\n"
           "*Synthetic data, not the real tape.* One true coin path; a vendor coin close carrying "
           "a transient timing error; two wrapper NAVs bleeding a planted drag plus "
           "premium/discount noise. The fund-vs-fund estimator must recover the planted spread "
           "with an interval clear of zero; the fund-vs-coin estimator, on the same world, must "
           "fail to. At `signal_strength=0` the wrappers are identical and the estimator must "
           "read zero."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from unstaked import data, strategy as st\n"
            "px, truth = data.synthetic_panel(signal_strength=1.0, seed=960)\n"
            "d = st.synthetic_detect(px)\n"
            "print(f\"planted spread {truth['spread_planted_pct']:.2f}%/yr\")\n"
            "print(f\"  fund-vs-fund {d['fund_vs_fund_pct']:+.2f}%/yr  \"\n"
            "      f\"CI [{d['fund_vs_fund_ci'][0]:+.2f}, {d['fund_vs_fund_ci'][1]:+.2f}]  \"\n"
            "      f\"monthly t {d['fund_vs_fund_monthly_t']:+.2f}\")\n"
            "print(f\"  fund-vs-coin {d['fund_vs_coin_pct']:+.2f}%/yr  \"\n"
            "      f\"CI [{d['fund_vs_coin_ci'][0]:+.2f}, {d['fund_vs_coin_ci'][1]:+.2f}]  \"\n"
            "      f\"-> {d['fund_vs_coin_halfwidth']/d['fund_vs_fund_halfwidth']:.0f}x wider\")\n"
            "nulls = np.array([st.synthetic_detect(\n"
            "    data.synthetic_panel(signal_strength=0.0, seed=960+s)[0])['fund_vs_fund_pct']\n"
            "    for s in range(8)])\n"
            "print(f\"  null x8: mean {nulls.mean():+.3f}%/yr (sd {nulls.std(ddof=1):.3f}), \"\n"
            "      f\"|est|>=0.5 in {(abs(nulls)>=0.5).sum()}/8\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The fund-vs-coin tracking difference asked for by the claim is "
           f"**{R['etha_td']:+.2f}%/yr** (ETHA) with a block-bootstrap CI of "
           f"**[{R['etha_lo']:+.2f}, {R['etha_hi']:+.2f}]** and a monthly *t* of {R['etha_mt']:+.2f} — "
           f"no signal, and none obtainable, since ETH-USD is an **unstaked** price from which "
           f"the reward is absent by construction. The same estimator against a same-close "
           f"benchmark delivers a genuinely robust number: **ETHE − ETHA = {R['ff_td']:+.2f}%/yr**, "
           f"CI **[{R['ff_lo']:+.2f}, {R['ff_hi']:+.2f}]**, monthly *t* = **{R['ff_mt']:+.2f}**, "
           f"recovering {R['ff_share_of_fee']}% of a documented {R['fee_spread']:.2f}%/yr fee gap "
           f"that lies inside the interval, with {R['ff_months_neg']}/{R['ff_months_n']} months "
           f"negative and the interval clear of zero at every block length. Two things the badge "
           f"is not allowed to hide: the i.i.d. daily *t* is only {R['ff_iid']:+.2f} (discarded "
           f"for the reason in §3, not ignored), and this is **2 of ~{R['cohort_n']} cohort "
           f"funds at the widest fee gap available**. Real on the fee, silent on the staking.\n"
           f"- **Tradability — Fragile.** Owning the cheap wrapper is worth the measured "
           f"{abs(R['ff_td']):.2f}%/yr ({R['ff_terminal_pct']:.1f}% of terminal wealth over "
           f"{R['ff_years']:.2f} years) against a documented {R['fee_spread']:.2f}%/yr fee gap, "
           f"for one trade and no borrow — but the evidence is 23 months of one asset and one "
           f"fee-extreme pair, and the spread has narrowed from "
           f"{R['era_e_ff']:+.2f} to {R['era_l_ff']:+.2f}%/yr. The levered expression dies above "
           f"~{R['borrow_be']} bps of borrow. The ~{R['stake']:.0f}%/yr staking yield is not "
           f"harvestable from inside any wrapper here.\n"
           f"- **Method, which is the durable output.** Report the resolution limit before the "
           f"point estimate — and report it on more than one ruler. Against a 24/7 coin it runs "
           f"±{R['res_min']:.1f} to ±{R['res_max']:.1f}%/yr depending on the block length, every "
           f"member of that range larger than the {R['stake']:.0f}%/yr effect; against a "
           f"same-close sibling fund it is ±{(R['ff_hi']-R['ff_lo'])/2:.2f}%/yr and ruler-stable. "
           f"The benchmark, not the estimator, decides whether the question can be answered."),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    for name, builder in (("01_for_the_curious", build_curious),
                          ("02_for_the_quants", build_quants)):
        nb = builder()
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
