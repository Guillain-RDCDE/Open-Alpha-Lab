"""Generate the two narrative notebooks for Study 932 (Trust Yield).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Real-tape headline numbers live in the single frozen ``R`` dict below, mirroring
docs/results.md. Cells that run live are either (a) the fast offline synthetic control —
never under a real-tape banner — or (b) an explicitly labelled recomputation from the
shared cache, which prints a clear "cache absent" line rather than silently substituting
synthetic numbers.
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
# 31 hardcoded pre-deal SPACs (unadjusted closes off the successor ticker) vs BIL,
# month-end signal, next-close execution, 15 bps one-way, redeem at trust.
# --------------------------------------------------------------------------- #
R = dict(
    asof="2026-06-30", fp="7d2c2f7b9919", fp_cash="b9dc0d8fa0bf",
    tape_start="2019-03-27", tape_end="2024-04-10",
    n_listed=31, n_window=31, n_traded=25, n_pos=203, hold_days=302,
    bad_prints=20, bad_name="CXAI",
    disc=1.47, ytr=4.03, exc=1.34, exc_ann=2.31, hit=93.1, t=17.87, t_hac=17.96,
    ci_lo=1.04, ci_hi=1.63, ci_neg=0.0,
    # The three ways to annualise. `exc_ann` (the mean of the per-position annualised
    # excess) over-weights short holds and is NOT a rate of return; `book_cagr` is.
    geo_ann=1.62, book_cagr=1.23, book_gross=1.39,
    one_n=25, one_exc=1.45, one_ann=1.61, one_t=7.46, one_lo=1.09, one_hi=1.83, one_hit=100,
    best_exc=10.99, best_lo=4.66, best_hi=18.02, best_t=9.12,
    # The headline is an identity: excess == (1+cash) x entry discount.
    ident_resid="3.3e-04", ident_corr=0.9992,
    # (a) the tape's own verdict on the assumed trust line, per shell
    anchor_med=0.68, anchor_above=14, anchor_within2=14, anchor_worst=-2.87,
    anchor_worst_name="ALTI", anchor_below_n=11, anchor_below_mean=-0.81,
    # (b) sell into that quote instead of redeeming: assumes NO trust payoff
    sell_exc=10.43, sell_t=8.49, sell_lo=3.96, sell_hi=17.70,
    sell_one_exc=10.99, sell_one_t=2.54, sell_one_lo=4.05, sell_one_hi=19.88, sell_one_hit=80,
    # (c) adversarial payoff: the worse of the assumed trust and that quote
    floor_exc=0.83, floor_t=9.29, floor_lo=0.28, floor_hi=1.28, floor_hit=78, floor_book=0.86,
    floor_one_exc=1.09, floor_one_t=5.40, floor_one_lo=0.70, floor_one_hi=1.47,
    # look-ahead: the horizon filter is built from the realised deal date
    la_n=218, la_exc=1.36, la_lo=1.06, la_hi=1.63,
    acc_sharpe=5.49, acc_cagr=1.23, acc_vol=0.22, acc_dd=-0.24, acc_t=13.25,
    mtm_sharpe=0.36, mtm_cagr=6.91, mtm_vol=37.34, mtm_dd=-41.20, mtm_t=0.95,
    mtm_ci_lo=-0.31, mtm_ci_hi=0.92, mtm_neg=13.8,
    era_a_n=124, era_a_s=24, era_a_disc=1.40, era_a_exc=1.25, era_a_ann=1.98,
    era_a_t=14.52, era_a_lo=0.98, era_a_hi=1.52, era_a_hit=94,
    era_b_n=79, era_b_s=9, era_b_disc=1.57, era_b_exc=1.47, era_b_ann=2.83,
    era_b_t=10.80, era_b_lo=0.79, era_b_hi=2.01, era_b_hit=92,
    # sweeps: (setting, n, excess per position, BOOK exCAGR, t[, hit])
    cost=((0, 1.49, 1.39, 19.88, 95), (15, 1.34, 1.23, 17.87, 93),
          (50, 0.98, 0.84, 13.16, 82), (100, 0.47, 0.30, 6.40, 65),
          (200, -0.52, -0.78, -7.19, 31)),
    trust=((9.90, 130, 0.83, 0.56, 10.51), (10.00, 203, 1.34, 1.23, 17.87),
           (10.10, 241, 2.05, 2.11, 25.58), (10.20, 251, 2.97, 3.19, 34.95)),
    buffer=((15, 1.31, 1.22), (30, 1.34, 1.23), (60, 1.41, 1.33), (90, 1.42, 1.38)),
    fee=((0, 1.34, 1.23), (10, 1.23, 1.08), (25, 1.03, 0.83)),
    guard_note="identical at +1.34% on 6% / 12% / 25% / off",
    sgov_exc=1.42, sgov_ann=2.75, sgov_t=16.91, sgov_n=179,
    drop_exc=1.40, drop_ann=2.40, drop_t=18.28, drop_n=189,
    peak_month="2022-08", peak_disc=2.03, peak_ytr=7.06, peak_bill=2.86,
    mania_month="2021-02", mania_disc=-9.44, mania_live=13,
    live_2021=16, live_2022=5, live_2024=1,
    syn_seeds=16, syn_pl_mean=1.938, syn_pl_sd=0.144, syn_pl_t=16, syn_pl_ci=16,
    syn_nl_mean=0.002, syn_nl_sd=1.585, syn_nl_t=6, syn_nl_ci=1,
)

BOOT = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
)

HEADER = f"""# Study 932 — Trust Yield 🏦

**A pre-deal SPAC quoted below its trust: a T-bill on sale, or a trap?**

A SPAC is a pot of Treasuries with a deadline. It raises about **$10 a share** into a
trust, and every public share carries a *redemption right* — hand the share back at the
deal vote (or at liquidation) and collect your pro-rata slice of that trust, plus the
interest it earned. The right does not depend on the deal failing, and it does not depend
on how you vote.

So a pre-deal SPAC quoted **below** trust is a Treasury bought at a discount. Annualise
that discount over the time left, and you get a yield *on top of* the bill yield — with
any deal upside thrown in for free.

We test the flat version of that trade on **{R['n_listed']} hardcoded 2019-2022-vintage
SPACs** ({R['tape_start']} → {R['tape_end']}) against **BIL**: each month-end, buy every
shell quoted under trust, execute at the **next** close, hold to the redemption deadline,
**redeem at trust**. 15 bps one-way, no shorting, one execution lag.

*Every real number below is the frozen headline from `docs/results.md` (fingerprint
`{R['fp']}`, as-of {R['asof']}). Live cells are labelled as such.*
"""


# --------------------------------------------------------------------------- #
# 01 — for the curious
# --------------------------------------------------------------------------- #
def build_curious():
    cells = [
        md(HEADER),
        md("## 1. What you are actually buying\n\n"
           "Forget the target company, the SPAC mania, the celebrity sponsor. Before a deal "
           "closes, the thing you own is a claim on a pile of short-dated US Treasuries and a "
           "contractual right to ask for it back on a known date. If the market sells you that "
           "claim for less than the pile is worth, the arithmetic does the rest.\n\n"
           "Here is what the tape actually offered, on average, when it was below trust:"),
        code(
            "R = %r\n"
            "print(f\"average discount to trust at entry : {R['disc']:+.2f}%%\")\n"
            "print(f\"average time left to redemption    : {R['hold_days']:.0f} days\")\n"
            "print(f\"=> implied annualised yield-to-redemption: {R['ytr']:+.2f}%%\")\n"
            "print(f\"what it actually paid, over cash   : {R['exc']:+.2f}%% per position\")\n"
            "print(f\"as a book, per year over T-bills   : {R['book']:+.2f}%%\")\n"
            "print(f\"positions that made money          : {R['hit']:.1f}%% of {R['n_pos']}\")"
            % (dict(disc=R["disc"], hold_days=R["hold_days"], ytr=R["ytr"], exc=R["exc"],
                    book=R["book_cagr"], hit=R["hit"], n_pos=R["n_pos"]),)
        ),
        md("## 2. The answer: yes, and it is small\n\n"
           f"Of the {R['n_listed']} shells on the list, **{R['n_traded']}** ever traded below "
           f"trust. Every single one of those {R['one_n']} paid — a **{R['one_hit']}%** hit rate "
           f"on the one-bet-per-name cut. But look at the size of the prize: about "
           f"**{R['book_cagr']:.1f}% a year over T-bills**, for money locked up for the best part "
           f"of a year in a shell company nobody was trading.\n\n"
           "> ⚠️ **The honest caveat, up front.** This study *pays itself the trust*: the model "
           "assumes a redeemed share hands back its accrued $10, because that is what the "
           "contract says. So the headline is closer to arithmetic than to a discovery — what "
           "the tape genuinely establishes is that the discount to that line was real and that "
           "**the line itself was where the market was** (on the day the redemption right "
           f"expired, {R['anchor_above']} of the {R['n_traded']} shells were quoted *at or "
           f"above* it, and the worst was only {abs(R['anchor_worst']):.1f}% under). Notebook 02 "
           "does the arithmetic in public.\n\n"
           "> 🔬 **For the quants.** The honest interval is a cluster bootstrap over the "
           f"{R['n_traded']} names, not a *t* on the {R['n_pos']} overlapping monthly entries: "
           f"95% CI **[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]** per position, zero negative "
           "resamples. The synthetic null in notebook 02 shows exactly why the naive *t* of "
           f"{R['t']:+.1f} must not be quoted."),
        md("## 3. Why the discount was there at all\n\n"
           "In 2021 there was no discount — there was a **premium**. Money piled into pre-deal "
           f"shells hoping for the next hot merger; in {R['mania_month']} the median shell on "
           f"this list traded **{abs(R['mania_disc']):.1f}% above** its trust. Buying a Treasury "
           "for eleven dollars is not an arbitrage.\n\n"
           "Then the mood broke and rates rose. Holders wanted out, there were no buyers, and "
           f"by **{R['peak_month']}** the median live shell was **{R['peak_disc']:+.1f}% below** "
           f"trust — an implied **{R['peak_ytr']:.1f}%** yield-to-redemption against a "
           f"**{R['peak_bill']:.1f}%** three-month bill. That is the whole trade, and it lasted "
           "about eighteen months."),
        md("## 4. The four things that could have gone wrong\n\n"
           "1. **The trust might not be $10.** It is the assumption the whole result rests on. "
           f"At $9.90 the edge more than halves ({R['trust'][0][3]:+.2f}% a year); at $10.20 it "
           f"triples. We do not know each shell's filed figure — we assume $10.00 and sweep it. "
           "The sharpest version of this test is to let the market overrule us shell by shell: "
           "pay the *worse* of the assumed trust and what the shell was actually quoted at on "
           f"the deadline. The edge survives that, at {R['floor_exc']:+.2f}% a position "
           f"({R['floor_book']:+.2f}% a year).\n"
           "2. **The deadline might move.** 2022-2023 was the era of serial extension votes. "
           "Each extension is *also* a redemption chance, so it shortens the realised wait — but "
           "it turns a dated bill into an open-ended one.\n"
           f"3. **The spread might eat it.** At 100 bps one-way the edge is down to "
           f"{R['cost'][3][2]:+.2f}% a year; at 200 bps it is **negative**. One to three cents "
           "on a $9.80 quote is not a fantasy in size.\n"
           f"4. **There might be nothing to buy.** Live pre-deal shells on this list: "
           f"{R['live_2021']} in early 2021, {R['live_2022']} by end-2022, {R['live_2024']} by "
           "2024. The market died."),
        md("## 5. Live check — the machinery is not inventing this (offline synthetic)\n\n"
           "Buying below a line and being paid that line is arithmetic, so the only null worth "
           "running is one where the **line is a fiction**: quotes wander with no anchor to any "
           "trust, and what you are paid at the end is simply the last quote. In that world the "
           "rule must earn nothing. Six worlds of each kind, run live below."),
        code(
            BOOT +
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import numpy as np\n"
            "from trust_yield import data, strategy as st\n"
            "for ss, label in [(1.0, 'trust put binds  '), (0.0, 'put is a fiction ')]:\n"
            "    outs = [st.synthetic_detect(*data.synthetic_panel(signal_strength=ss, seed=s,\n"
            "                                                     n_days=700)[:3], n_boot=600)\n"
            "            for s in range(932, 938)]\n"
            "    m = np.array([o['mean_excess'] for o in outs])\n"
            "    fires = sum(1 for o in outs if o['ci_low'] > 0)\n"
            "    print(f\"{label}: mean excess {m.mean():+.2%}  \"\n"
            "          f\"(worlds where the edge is significant: {fires}/6)\")\n"
            "print('\\n(synthetic, not the tape — any single null world can fire by luck; '\n"
            "      'notebook 02 runs the rate over 16)')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The discount existed: **{R['exc']:+.2f}%** "
           f"per {R['hold_days']:.0f}-day position, cluster-bootstrap CI "
           f"**[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]** across "
           f"{R['n_traded']} shells with no negative resamples, positive in both rate eras. "
           "That number is an **identity** — we pay ourselves the trust by assumption — so the "
           "stamp leans on the parts that are not assumed: the deadline-day quote sat at or "
           f"above the assumed trust line in {R['anchor_above']}/{R['n_traded']} shells and "
           f"never more than {abs(R['anchor_worst']):.1f}% under it, and paying the *worse* of "
           f"the two still leaves {R['floor_exc']:+.2f}% a position. **Survivorship:** this is a "
           "list of shells whose successor ticker still trades and which never did a post-deal "
           "reverse split, so the *count of chances* is flattered — though the payoff itself "
           "(the trust) is what a liquidation pays too.\n"
           f"- **Tradability — Fragile.** {R['book_cagr']:+.1f}% a year over bills as a book, for "
           f"a {R['hold_days']:.0f}-day lock-up in a shell trading a few hundred thousand dollars "
           f"a day. It dies at 200 bps of friction, it more than halves if the trust was $9.90, "
           "it assumes a deadline that sponsors kept moving, the broker's fee for filing the "
           "redemption is not modelled at all, and the opportunity set collapsed from "
           f"{R['live_2021']} live shells to {R['live_2024']}. A real mechanism you could not "
           "have sized, in a market that no longer exists."),
    ]
    nb = new_notebook()
    nb["cells"] = cells
    return nb


# --------------------------------------------------------------------------- #
# 02 — for the quants
# --------------------------------------------------------------------------- #
def build_quants():
    cost_rows = "\n".join(
        f"print(f\"{c:>4} bps : excess {e:+.2f}%/position  book {a:+.2f}%/yr  "
        f"t {t:+.2f}  hit {h:.0f}%\")"
        for c, e, a, t, h in R["cost"])
    trust_rows = "\n".join(
        f"print(f\"${v:.2f} : n={n:3d}  excess {e:+.2f}%/position  book {a:+.2f}%/yr  "
        f"t {t:+.2f}\")"
        for v, n, e, a, t in R["trust"])
    buf_rows = "\n".join(
        f"print(f\"{b:>3}d : excess {e:+.2f}%/position  book {a:+.2f}%/yr\")"
        for b, e, a in R["buffer"])
    fee_rows = "\n".join(
        f"print(f\"{f:>3}bp : excess {e:+.2f}%/position  book {a:+.2f}%/yr\")"
        for f, e, a in R["fee"])

    cells = [
        md("# Study 932 — Trust Yield — the teardown\n\n"
           "The position-level excess over cash, the cluster bootstrap over shells, the "
           "one-bet-per-name cut, the two daily books, the era cut, four assumption sweeps, "
           "two cross-checks, the yield path, and the live synthetic control that shows why the "
           f"naive *t* must not be quoted. Real numbers are frozen from `docs/results.md` "
           f"(fingerprint `{R['fp']}`, cash `{R['fp_cash']}`, as-of {R['asof']})."),
        code("R = %r" % (R,)),
        md("## Construction\n\n"
           "- **Tape.** Pre-deal SPAC quotes are **unadjusted** closes read off each SPAC's "
           "*successor* ticker (Yahoo carries the pre-deal history forward). Unadjusted is "
           "deliberate: a $10 trust is a dollar quantity, and a post-deal reverse split would "
           "rescale the pre-deal tape into nonsense. Names that split post-deal are therefore "
           "**absent** — a documented selection.\n"
           "- **Trust path (PROXY).** $10.00 at IPO accreting at BIL's total return, less an "
           "optional fee drag. Swept.\n"
           "- **Deadline (ASSUMPTION).** Hardcoded deal close **− 30 days**, because the "
           "redemption election is due before the vote and the vote before the close. Swept. "
           "Without the buffer several shells de-anchor violently in the final fortnight — "
           "which is the trust put expiring, exactly on cue.\n"
           "- **Execution.** Signal on the month-end close, buy at the **next** close. One lag. "
           "15 bps one-way × NAV at entry; redemption is at trust and free. No shorts, so no "
           "borrow leg.\n"
           f"- **Cleaning.** {R['bad_prints']} pre-deal quotes below 60% of trust (all in "
           f"{R['bad_name']}, a 2023 vendor artefact) are replaced by the previous good quote — "
           "inside the pre-deal window only. Dropping that name entirely *raises* the headline."),
        md("## The headline"),
        code(
            "print(f\"{R['n_pos']} positions across {R['n_traded']} SPACs "
            "(of {R['n_listed']} listed); mean hold {R['hold_days']:.0f} days\")\n"
            "print(f\"mean discount at entry      {R['disc']:+.2f}%\")\n"
            "print(f\"mean implied YTR at entry   {R['ytr']:+.2f}%\")\n"
            "print(f\"mean excess over cash       {R['exc']:+.2f}% per position\")\n"
            "print(f\"hit rate {R['hit']:.1f}%   one-sample t {R['t']:+.2f}   HAC t {R['t_hac']:+.2f}\")\n"
            "print(f\"cluster bootstrap over {R['n_traded']} shells: "
            "95% CI [{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%], share<0 {R['ci_neg']:.1f}%\")\n"
            "print()\n"
            "print(f\"one position per SPAC (n={R['one_n']}): {R['one_exc']:+.2f}%, "
            "t {R['one_t']:+.2f}, "
            "CI [{R['one_lo']:+.2f}%, {R['one_hi']:+.2f}%], hit {R['one_hit']}%\")\n"
            "print(f\"redeem-or-sell upper bound : {R['best_exc']:+.2f}% "
            "CI [{R['best_lo']:+.2f}%, {R['best_hi']:+.2f}%] t {R['best_t']:+.2f}\")"
        ),
        md("> 💡 **In plain words.** Twenty-five shells, every one of them a winner, for about "
           "a percent and a third apiece. The 'redeem-or-sell' line is what you would have made "
           "if you had been allowed to look at the deadline-day quote and take the better of it "
           "and the trust — eight times more. We do **not** claim that number: the redemption "
           "election is filed days in advance, so the rule redeems unconditionally and gives up "
           "every deal premium on offer. The headline is the conservative branch."),
        md("## Now read that table again: it is an identity\n\n"
           "The payoff above is **imposed** — the model hands a redeemed share its accrued "
           "trust — and the benchmark is the same cash leg the trust is assumed to accrue at. "
           "Put those two together and the position excess collapses to\n\n"
           "```\n"
           "excess  =  (1 + cash_ret) x (trust at signal / cost-loaded entry price - 1)\n"
           "```\n\n"
           "i.e. **to the entry discount**. The cell below checks that on the real positions."),
        code(
            f"print(f\"max |excess - (1+cash) x entry discount| over all {{R['n_pos']}} \"\n"
            f"      f\"positions : {{R['ident_resid']}}\")\n"
            f"print(f\"correlation of the excess with the entry discount    : "
            f"{{R['ident_corr']:.4f}}\")\n"
            "print()\n"
            "print('=> the +%.2f%%, the %.1f%% hit rate, the t of %+.1f and the cluster CI are'\n"
            "      % (R['exc'], R['hit'], R['t']))\n"
            "print('   all statistics about THE ENTRY DISCOUNT - about whether a discount')\n"
            "print('   selected on Monday was still there on Tuesday. None of them is')\n"
            "print('   evidence that the redemption paid: that is an assumption, not a')\n"
            "print('   measurement. The one-per-name t of %+.2f is the same identity with'\n"
            "      % R['one_t'])\n"
            "print('   fewer observations.')"
        ),
        md("That is not a scandal — a redemption right *is* a contractual identity, and this "
           "desk has stamped mechanical identities Real before. But it means the Signal stamp "
           "has to be earned somewhere the model is not doing the talking. Three places:\n\n"
           "**(a) Did the market agree with our trust line?** On the day the redemption right "
           "expired, where was the shell actually quoted, against the $10.00-accreted-at-BIL "
           "line we assumed?"),
        code(
            "print(f\"median gap (quote - assumed trust)      : {R['anchor_med']:+.2f}%\")\n"
            "print(f\"shells quoting at or above the line     : {R['anchor_above']}/{R['n_traded']}\")\n"
            "print(f\"shells within +/-2% of it               : {R['anchor_within2']}/{R['n_traded']}\")\n"
            "print(f\"worst shell                             : {R['anchor_worst']:+.2f}% "
            "({R['anchor_worst_name']})\")\n"
            "print(f\"mean shortfall on the {R['anchor_below_n']} shells below : "
            "{R['anchor_below_mean']:+.2f}%\")"
        ),
        md("No shell's quote fell more than **2.9%** under the assumed line. The trust value we "
           "guessed is roughly where the market itself was when the put expired.\n\n"
           "**(b) The assumption-free version.** *Sell* into that deadline quote instead of "
           "redeeming, so the trust never enters the payoff at all:"),
        code(
            "print(f\"all positions : {R['sell_exc']:+.2f}%  t {R['sell_t']:+.2f}  "
            "CI [{R['sell_lo']:+.2f}%, {R['sell_hi']:+.2f}%]\")\n"
            "print(f\"one per shell : {R['sell_one_exc']:+.2f}%  t {R['sell_one_t']:+.2f}  "
            "CI [{R['sell_one_lo']:+.2f}%, {R['sell_one_hi']:+.2f}%]  hit {R['sell_one_hit']}%\")\n"
            "print()\n"
            "print('clears |t| >= 2 on the near-independent cut with NO trust assumption -')\n"
            "print('but it is a different, far wilder trade (its return is de-SPAC hype:')\n"
            "print('one shell quoted +100% over trust on its deadline). Corroboration,')\n"
            "print('not the headline.')"
        ),
        md("**(c) The adversarial payoff.** Let the tape veto our assumption shell by shell: "
           "pay the **worse** of the assumed accrued trust and the deadline-day quote "
           "(`market_floor=True`). Deliberately too harsh — the redemption right pays trust "
           "whatever the screen says — which is the point of a floor:"),
        code(
            "print(f\"all positions : {R['floor_exc']:+.2f}%  t {R['floor_t']:+.2f}  "
            "CI [{R['floor_lo']:+.2f}%, {R['floor_hi']:+.2f}%]  hit {R['floor_hit']}%  "
            "book {R['floor_book']:+.2f}%/yr\")\n"
            "print(f\"one per shell : {R['floor_one_exc']:+.2f}%  t {R['floor_one_t']:+.2f}  "
            "CI [{R['floor_one_lo']:+.2f}%, {R['floor_one_hi']:+.2f}%]\")\n"
            "print()\n"
            "print('the edge survives the harshest haircut available on its own')\n"
            "print('load-bearing assumption, at roughly two thirds of its size.')"
        ),
        md("## Annualising it honestly\n\n"
           "Three numbers, only one of which is a rate of return:"),
        code(
            "print(f\"mean of the per-position annualised excess : {R['exc_ann']:+.2f}%  \"\n"
            "      f\"<- over-weights short holds; NOT a rate of return\")\n"
            "print(f\"the mean position over its {R['hold_days']:.0f}-day hold      : "
            "{R['geo_ann']:+.2f}%\")\n"
            "print(f\"THE BOOK (equal-weighted, net of cost)     : {R['book_cagr']:+.2f}%\")\n"
            "print()\n"
            "print('Drop the 30-day minimum horizon and the first number goes to +112%,')\n"
            "print('which is all you need to know about quoting it. The book is the')\n"
            "print('bankable figure and it is what the Tradability card uses.')"
        ),
        md("## Look-ahead, named\n\n"
           "The deadline and the 30-730-day horizon filter are built from the **realised** deal "
           "date, which nobody knew at entry. (A real holder would have redeemed against the "
           "*charter* deadline, which was known ex ante.) Lifting the horizon cap to 3000 days "
           "moves nothing:"),
        code(
            "print(f\"max_days= 730 : n={R['n_pos']}  excess {R['exc']:+.2f}%  "
            "CI [{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]\")\n"
            "print(f\"max_days=3000 : n={R['la_n']}  excess {R['la_exc']:+.2f}%  "
            "CI [{R['la_lo']:+.2f}%, {R['la_hi']:+.2f}%]\")\n"
            "print('\\nthe look-ahead is present and inert.')"
        ),
        md("## Why the naive *t* is not the number\n\n"
           "Monthly entries in the same shell are the same bet on the same deadline: the same "
           "trust, the same date, largely the same discount. The independent unit is the "
           "**name**. The cluster bootstrap resamples whole shells; the synthetic null below "
           "quantifies what the naive interval costs."),
        md("## The two daily books (excess-of-cash)"),
        code(
            "print(f\"accrual (hold-to-redemption): exSharpe {R['acc_sharpe']:+.2f}  "
            "exCAGR {R['acc_cagr']:+.2f}%  vol {R['acc_vol']:.2f}%  "
            "maxDD {R['acc_dd']:+.2f}%  HAC t {R['acc_t']:+.2f}\")\n"
            "print(f\"mark-to-market             : exSharpe {R['mtm_sharpe']:+.2f}  "
            "exCAGR {R['mtm_cagr']:+.2f}%  vol {R['mtm_vol']:.2f}%  "
            "maxDD {R['mtm_dd']:+.2f}%  HAC t {R['mtm_t']:+.2f}\")\n"
            "print(f\"mark-to-market Sharpe 95% CI [{R['mtm_ci_lo']:+.2f}, {R['mtm_ci_hi']:+.2f}]  "
            "share<0 {R['mtm_neg']:.1f}%\")"
        ),
        md(f"The accrual Sharpe of **{R['acc_sharpe']:+.2f}** is an artefact of the "
           "construction, not a claim: once a position is on, the pull to trust is "
           "deterministic, so the only variance left is the changing mix of open positions. "
           f"The mark-to-market book is the honest path, and its Sharpe CI "
           f"**[{R['mtm_ci_lo']:+.2f}, {R['mtm_ci_hi']:+.2f}]** straddles zero. Most of its "
           f"{R['mtm_vol']:.0f}% vol and {abs(R['mtm_dd']):.0f}% drawdown is the *modelled "
           "forced redemption*: shells quoting a deal premium on the deadline day get marked "
           "down to trust in a single print. That is forgone upside recognised at once, not a "
           "loss — and it is the cost of the conservative branch we chose."),
        md("## Era cut (entry year, split 2022)"),
        code(
            "print(f\"2019-2021 (zero rates) : n={R['era_a_n']:3d} ({R['era_a_s']} shells)  "
            "disc {R['era_a_disc']:+.2f}%  excess {R['era_a_exc']:+.2f}% ({R['era_a_ann']:+.2f}% ann)  "
            "t {R['era_a_t']:+.2f}  CI [{R['era_a_lo']:+.2f}%, {R['era_a_hi']:+.2f}%]  hit {R['era_a_hit']}%\")\n"
            "print(f\"2022-2024 (hiking)     : n={R['era_b_n']:3d} ({R['era_b_s']} shells)  "
            "disc {R['era_b_disc']:+.2f}%  excess {R['era_b_exc']:+.2f}% ({R['era_b_ann']:+.2f}% ann)  "
            "t {R['era_b_t']:+.2f}  CI [{R['era_b_lo']:+.2f}%, {R['era_b_hi']:+.2f}%]  hit {R['era_b_hit']}%\")"
        ),
        md("Positive in both, wider and faster in the hiking era — but the late era rests on "
           f"**{R['era_b_s']}** shells. The opportunity set was already collapsing while the "
           "yield was at its best."),
        md("## Sweep 1 — cost (one-way on NAV at entry; redemption free)"),
        code(cost_rows),
        md("## Sweep 2 — the trust level, the load-bearing assumption"),
        code(trust_rows),
        md("Everything scales with a number we assumed rather than filed: a dime of trust moves "
           "the result by more than the result itself. $10.00 is the standard 2020-2021 trust "
           "funding, but over-funded shells at $10.10-$10.20 existed, and extension votes "
           f"sometimes topped the trust up. **$9.90 is the honest floor** and it still clears "
           f"zero ({R['trust'][0][3]:+.2f}%/yr as a book) — and the market's own shell-by-shell "
           f"answer, the `market_floor` cut above, lands at {R['floor_book']:+.2f}%/yr. That is "
           "what keeps the Signal stamp green."),
        md("## Sweeps 3 and 4 — redemption buffer and trust fee drag"),
        code(buf_rows + "\nprint()\n" + fee_rows +
             "\nprint()\nprint('deep-quote guard: %s')" % R["guard_note"]),
        md("The guard never binds: once the vendor artefacts are cleaned, **no** entry in the "
           "whole sample implied a discount deeper than 6%. This was always a small, patient "
           "trade, never a distressed one."),
        md("## Cross-checks"),
        code(
            "print(f\"SGOV instead of BIL as cash/accrual leg : {R['sgov_exc']:+.2f}% "
            "({R['sgov_ann']:+.2f}% ann)  t {R['sgov_t']:+.2f}  n={R['sgov_n']}\")\n"
            "print(f\"drop the print-cleaned name ({R['bad_name']})      : {R['drop_exc']:+.2f}% "
            "({R['drop_ann']:+.2f}% ann)  t {R['drop_t']:+.2f}  n={R['drop_n']}\")"
        ),
        md("## The yield path — when the trade existed"),
        code(
            "print(f\"{R['mania_month']} (the mania): median shell {abs(R['mania_disc']):.1f}% \"\n"
            "      f\"ABOVE trust across {R['mania_live']} live shells — no trade\")\n"
            "print(f\"{R['peak_month']} (the peak) : median discount {R['peak_disc']:+.2f}%, \"\n"
            "      f\"implied YTR {R['peak_ytr']:+.2f}% vs a {R['peak_bill']:.2f}% 3m bill\")\n"
            "print(f\"live pre-deal shells: {R['live_2021']} (early 2021) -> \"\n"
            "      f\"{R['live_2022']} (end 2022) -> {R['live_2024']} (2024)\")"
        ),
        md("## Live synthetic control — the plant, the null, and the cost of the naive *t*\n\n"
           "**Offline synthetic, not the tape.** The null cannot be a smaller discount — buying "
           "below a line and being paid that line is arithmetic. So the null breaks the "
           "*mechanism*: quotes are an unanchored martingale accruing at the cash rate, and the "
           "terminal payoff is whatever the last quote happens to be. Run live over 16 seeds, "
           "scoring both the naive position-level *t* and the cluster bootstrap."),
        code(
            BOOT +
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import numpy as np\n"
            "from trust_yield import data, strategy as st\n"
            "for ss, label in [(1.0, 'put binds      '), (0.0, 'put is fiction ')]:\n"
            "    outs = [st.synthetic_detect(*data.synthetic_panel(signal_strength=ss, seed=s,\n"
            "                                                     n_days=700)[:3],\n"
            "                                n_boot=800)\n"
            "            for s in range(932, 948)]\n"
            "    m = np.array([o['mean_excess'] for o in outs])\n"
            "    t = np.array([o['t_pos'] for o in outs])\n"
            "    ci = sum(1 for o in outs if o['ci_low'] > 0)\n"
            "    print(f\"{label}: mean excess {m.mean():+.3%} (sd {m.std(ddof=1):.3%})  \"\n"
            "          f\"naive t>2 on {int((t>2).sum()):2d}/16   cluster CI>0 on {ci:2d}/16\")"
        ),
        md(f"The frozen 24-shell version in `docs/results.md`: the plant is recovered "
           f"**{R['syn_pl_ci']}/{R['syn_seeds']}** times (mean {R['syn_pl_mean']:+.3f}%), the "
           f"null is dead-centred (mean {R['syn_nl_mean']:+.3f}%, sd {R['syn_nl_sd']:.3f}%) — "
           f"and the naive position-level *t* false-fires on **{R['syn_nl_t']}/{R['syn_seeds']}** "
           f"null panels where the cluster bootstrap fires on **{R['syn_nl_ci']}/{R['syn_seeds']}**, "
           "right at its nominal 5%. That is the whole argument for the interval this study quotes."),
        md("## Live recomputation from the shared cache (labelled)\n\n"
           "**Real tape.** This cell recomputes the headline from `studies/_cache` and checks it "
           "against the frozen dict. If the cache is absent it says so and computes nothing — it "
           "never substitutes synthetic numbers under this banner."),
        code(
            BOOT +
            "import warnings; warnings.filterwarnings('ignore')\n"
            "from trust_yield import data, strategy as st\n"
            "if not data.have_real():\n"
            "    print('shared _cache absent — real-tape recomputation skipped '\n"
            "          '(run examples/verify.py --fetch to populate it). No numbers shown.')\n"
            "else:\n"
            "    px, bad = data.clean_quotes(data.load_spac_closes())\n"
            "    cash = data.load_cash()['BIL'].dropna()\n"
            "    live = st.race(px, cash, cost_bps=15.0)\n"
            "    live_book = st.summary(st.portfolio_daily(live['positions'], px, cash,\n"
            "                                              mode='accrual'))\n"
            "    print(f\"live : {live['n_pos']} positions / {live['n_spacs']} shells   \"\n"
            "          f\"excess {live['mean_excess']:+.2%}   book {live_book['cagr']:+.2%}/yr   \"\n"
            "          f\"CI [{live['boot']['ci_low']:+.2%}, {live['boot']['ci_high']:+.2%}]\")\n"
            "    print(f\"frozen: {R['n_pos']} positions / {R['n_traded']} shells   \"\n"
            "          f\"excess {R['exc']/100:+.2%}   book {R['book_cagr']/100:+.2%}/yr   \"\n"
            "          f\"CI [{R['ci_lo']/100:+.2%}, {R['ci_hi']/100:+.2%}]\")\n"
            "    print(f\"identity residual: {live['identity']['max_abs_residual']:.1e}  \"\n"
            "          f\"corr with entry discount {live['identity']['corr_with_discount']:.4f}\")\n"
            "    print('bad prints cleaned:', bad)\n"
            "    print('fingerprint:', data.fingerprint(px), '(frozen', R['fp'] + ')')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** Mean excess over cash of **{R['exc']:+.2f}%** per "
           f"{R['hold_days']:.0f}-day position, cluster CI "
           f"**[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]** over {R['n_traded']} shells with zero "
           f"negative resamples, positive in both eras, robust to the buffer, the fee drag and "
           f"the quote guard. **That headline is an identity** — the payoff is imposed, so its "
           f"*t* of {R['t']:+.1f} (and the one-per-name {R['one_t']:+.2f}) describe the entry "
           "discount, not the redemption. The stamp rests on the three unassumed reads: the "
           f"deadline quote at or above the assumed trust in {R['anchor_above']}/{R['n_traded']} "
           f"shells and never more than {abs(R['anchor_worst']):.1f}% below; the sell-at-quote "
           f"version clearing *t* = {R['sell_one_t']:+.2f} one-per-name with no trust assumption "
           f"at all; and the `market_floor` payoff still paying {R['floor_exc']:+.2f}% "
           f"(CI [{R['floor_lo']:+.2f}%, {R['floor_hi']:+.2f}%]). The mechanism is visible in the "
           f"path too: implied YTR of {R['peak_ytr']:.1f}% against a {R['peak_bill']:.1f}% bill at "
           "the peak. **Survivorship named:** successor-ticker, no-reverse-split shells only, so "
           "liquidations and split names are absent and the opportunity count is flattered; the "
           "payoff itself is unaffected, since a liquidation pays the trust too.\n"
           f"- **Tradability — Fragile.** {R['book_cagr']:+.1f}% a year over bills **as a book** "
           f"(the {R['exc_ann']:+.2f}% mean-of-annualised is not a rate of return) for a "
           f"{R['hold_days']:.0f}-day lock-up in a $200-300m shell trading a few hundred thousand "
           f"dollars a day; dead at 200 bps of friction ({R['cost'][4][2]:+.2f}%/yr); more than "
           f"halved if the trust was $9.90 ({R['trust'][0][3]:+.2f}%/yr); the broker's fee for "
           "filing the redemption election is not modelled at all; dependent on a deadline "
           f"sponsors kept moving; and gone — {R['live_2021']} live "
           f"shells in 2021, {R['live_2024']} by 2024. A sound mechanism you could not size, in a "
           "market that closed behind it."),
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
