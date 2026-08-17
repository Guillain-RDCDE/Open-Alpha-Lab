"""Generate the two narrative notebooks for Study 928 (Odd-Lot Priority).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
synthetic control, which is clearly labelled as synthetic and never sits under a real-tape
banner.
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


# Frozen real-tape headline — mirror of docs/results.md. 180 SC TO-I modified Dutch
# auction self-tenders filed 2010-2025, 178 after the recycled-symbol exclusion, 167 usable
# windows, SPY market leg, BIL cash leg, total-return closes, as-of 2026-06-30.
R = dict(
    start="2010-01-28", end="2025-11-21", n_events=167, n_listed=178, n_filed=180,
    n_tickers=128, n_ticker_ids=129,
    fp="4f4dc9efc748", per_year=10.6,
    premium=13.0, proration=0.35, expiry_td=21, fee=30.0, position=5000.0,
    cost_bps=15.0, borrow_bps=40.0,
    runup=7.41, runup_med=5.94, runup_t=9.45, runup_hit=0.80,
    runup_abn=6.87, runup_abn_t=9.06,
    drift=1.44, drift_t=1.75,
    give=-0.50, give_t=-1.17, give_hit=0.47,
    perm=8.63, perm_abn=6.23, perm_abn_t=5.24,
    odd=5.32, odd_t=7.56,
    hedged=3.52, hedged_t=4.32, hedged_hit=0.70, hedged_lo=1.95, hedged_hi=5.05,
    round_lot=1.93, round_t=3.10,
    priority=3.39, priority_t=4.33,
    # the value of priority is NOT a tape fact: it moves one-for-one with the assumed
    # premium and crosses zero at 7.4%
    prio5=-1.49, prio5_t=-1.95, prio9=0.95, prio9_t=1.23,
    prio17=5.83, prio17_t=7.34, prio25=10.72, prio25_t=13.12,
    prio_zero=7.4, prio_at_be=1.67, prio_at_be_t=2.15,
    share_post_above_clear=0.32, share_runup_gt_premium=0.20,
    be_mean=10.17, be_med=7.93, be_p75=14.56, be_share_above=0.30,
    prem5=-3.99, prem5_t=-5.18, prem9=-0.23, prem9_t=-0.30,
    prem17=7.27, prem17_t=8.68, prem25=14.78, prem25_t=16.72,
    era_e_n=74, era_e_runup=5.29, era_e_hedged=5.20, era_e_t=7.13, era_e_be=7.89,
    era_l_n=93, era_l_runup=9.10, era_l_hedged=2.19, era_l_t=1.72, era_l_be=11.99,
    # the CONTRAST itself, Welch t on late - early (two point estimates are not a contrast)
    d_runup=3.82, d_runup_t=2.63, d_be=4.10, d_be_t=2.49,
    d_hedged=-3.01, d_hedged_t=-1.99, d_give=-1.39, d_give_t=-1.77,
    d_priority=0.16, d_priority_t=0.11,
    pro15=4.44, pro35=3.39, pro75=1.31, pro100=0.00,
    fee0=4.57, fee50=2.37, fee50_t=2.91,
    # the bps touch is charged 3x per hedged round trip; the list is a third micro-cap
    c100=0.97, c100_t=1.19, c100_be=12.91, c100_fee0=1.57, c100_fee0_t=1.93,
    cap1000=1.12, cap1000_t=1.37, cap1000_yr=118, cap5000_yr=1859, cap9900_yr=3990,
    dollar_event=176,
    borrow300=3.30, borrow300_t=4.05,
    exp40=2.42, exp40_t=2.52, anchor21=2.15, anchor21_t=1.98, anchor21_be=22.62,
    cal_days=4000, cal_start="2010-02-01", cal_end="2025-12-23",
    cal_invested=51.9, cal_live=1.61, cal_vol=65.6,
    sh_gross=0.787, sh_net=0.715, sh_spy=0.799, sh_adv=-0.084, sh_t=2.09,
    syn_pl_runup=466, syn_pl_runup_t=10.42, syn_pl_give=-387, syn_pl_give_t=-9.93,
    syn_nl_runup=-44, syn_nl_runup_t=-1.04, syn_nl_give=6, syn_nl_give_t=0.14,
    syn_nl8_runup=43, syn_nl8_sd=70, syn_nl8_fire=1,
)


HEADER = f"""# Study 928 — Odd-Lot Priority

**The odd-lot holder is filled first and in full. Is that a real retail-only edge?**

Almost every US issuer self-tender carries an **odd-lot priority** clause: tender fewer
than 100 shares — all of them — and you are bought *first and in full*, ahead of the
proration that hits every round-lot holder. The forum conclusion writes itself: buy 99
shares, tender, collect the premium.

We test it on the **same {R['n_listed']} modified-Dutch-auction self-tenders** as Study 927
({R['start']} → {R['end']}, {R['n_events']} fully covered windows, {R['n_tickers']} of
{R['n_ticker_ids']} issuer tapes recoverable, plus **SPY** and **BIL**), daily total-return
closes. The offer is public at the close of day *t*; the odd lot is bought at the close of
*t+1* — the study's **one** execution lag.

*The clearing price of a tender is not in a price series.* Four inputs are declared
**assumptions** and swept end to end: the clearing premium ({R['premium']:.0f}% over the
pre-announcement close), the offer length ({R['expiry_td']} trading days), the round-lot
proration fill ({R['proration']:.2f}) and the flat broker corporate-action fee
(${R['fee']:.0f} on a ${R['position']:,.0f} lot). Everything else is tape — and where a
number below leans on an assumption, this notebook says so rather than letting the *t*
speak for the tape.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
live cells run the offline synthetic control. As-of 2026-06-30.*
"""

SYN_CELL = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "from odd_lot import data, strategy as st\n"
    "# SYNTHETIC tape only - not the real tender list.\n"
    "panel_p, ev_p, truth = data.synthetic_panel(n_events=90, signal_strength=1.0, seed=928)\n"
    "panel_0, ev_0, _     = data.synthetic_panel(n_events=90, signal_strength=0.0, seed=928)\n"
    "pl = st.synthetic_detect(panel_p, ev_p)\n"
    "nl = st.synthetic_detect(panel_0, ev_0)\n"
    "print('planted %+d bps run-up / %+d bps give-back ->  recovered %+.0f bps (t=%+.2f) / '\n"
    "      '%+.0f bps (t=%+.2f)'\n"
    "      % (truth['runup_planted_bps'], -truth['giveback_planted_bps'],\n"
    "         pl['runup_bps'], pl['runup_t'], pl['giveback_bps'], pl['giveback_t']))\n"
    "print('null (nothing planted)            ->  recovered %+.0f bps (t=%+.2f) / '\n"
    "      '%+.0f bps (t=%+.2f)'\n"
    "      % (nl['runup_bps'], nl['runup_t'], nl['giveback_bps'], nl['giveback_t']))"
)


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The trade, as the forums tell it\n\n"
           "A company offers to buy back its own shares at a premium. If everyone tenders, "
           "each holder is cut back *pro rata* — you offer 1,000 shares and they take 350. "
           "But the fine print says a holder of **fewer than 100 shares** who tenders all of "
           "them is bought **first and in full**. So: buy 99 shares the morning after the "
           "announcement, tender them, take the premium. Free money — allegedly."),
        md("## 2. The first problem: you are not the first buyer\n\n"
           f"The offer is public at the close of the announcement day. The earliest an "
           f"outside buyer can act is the *next* close. By then the market has already "
           f"taken **{R['runup']:.2f}%** of the premium "
           f"(**{R['runup_abn']:.2f}%** after subtracting SPY) — with a *t* of "
           f"**{R['runup_abn_t']:+.2f}**, this is about as certain as anything on this desk "
           f"gets. In **{R['share_runup_gt_premium']:.0%}** of events the pop *alone* is "
           f"bigger than the entire assumed {R['premium']:.0f}% premium."),
        code(
            "R = dict(runup=%r, runup_abn=%r, runup_abn_t=%r, premium=%r,\n"
            "         share=%r, hedged=%r, hedged_t=%r)\n"
            "print('premium assumed by the study : %%5.2f%%%%' %% R['premium'])\n"
            "print('taken by the pop before you can buy: %%5.2f%%%% (abnormal %%.2f%%%%, "
            "t=%%+.2f)'\n"
            "      %% (R['runup'], R['runup_abn'], R['runup_abn_t']))\n"
            "print('left for the odd-lot buyer, hedged and costed: %%5.2f%%%% (t=%%+.2f)'\n"
            "      %% (R['hedged'], R['hedged_t']))\n"
            "print('events where the pop alone beats the whole premium: %%.0f%%%%' "
            "%% (R['share']*100))"
            % (R["runup"], R["runup_abn"], R["runup_abn_t"], R["premium"],
               R["share_runup_gt_premium"], R["hedged"], R["hedged_t"])
        ),
        md("## 3. The second problem: priority is only worth the give-back\n\n"
           "Being filled first only matters if the tender price beats what you could have "
           "sold at afterwards. If the stock simply *keeps* the premium, the round-lot "
           "holder's un-filled stub is worth as much as your cash — and priority bought you "
           "nothing.\n\n"
           f"On this tape the post-expiry give-back is **{R['give']:+.2f}%** with *t* = "
           f"**{R['give_t']:+.2f}** and a median of exactly **0.00%** — statistically "
           f"absent. The stock keeps the premium: **{R['perm_abn']:+.2f}%** abnormal, "
           f"permanently. In **{R['share_post_above_clear']:.0%}** of events the post-expiry "
           f"price is *above* the assumed tender price, so priority actively cost you: it "
           f"forced you out of a stock that kept rising."),
        md("## 4. Careful: 'what priority is worth' is a number we assumed\n\n"
           f"It is tempting to read **{R['priority']:+.2f}%** per event (*t* = "
           f"{R['priority_t']:+.2f}) as the measured value of the entitlement. It is not. "
           f"Priority pays you the *tender* price instead of the *post-expiry* price, and we "
           f"only measured one of those two. Change the premium we assumed and the whole "
           f"number moves with it:\n\n"
           f"| assumed premium | 5% | 9% | **13%** | 17% | 25% |\n"
           f"|---|--:|--:|--:|--:|--:|\n"
           f"| value of priority | **{R['prio5']:+.2f}%** | {R['prio9']:+.2f}% | "
           f"**{R['priority']:+.2f}%** | {R['prio17']:+.2f}% | {R['prio25']:+.2f}% |\n\n"
           f"It crosses zero at **{R['prio_zero']:.1f}%** and is *negative* below that. So "
           f"the honest version of the sentence is: priority can only ever pay the gap "
           f"between the tender price and the post-expiry price — and section 3 just showed "
           f"that gap is missing."),
        md("## 5. The one number that needs no premium: how big would it have to be?\n\n"
           f"Rather than pretend we know each offer's clearing price, we ask the tape the "
           f"question backwards: **how large would the premium have to be** for the costed, "
           f"market-hedged odd-lot round trip to merely match the market?\n\n"
           f"**{R['be_mean']:.2f}%** on average (median {R['be_med']:.2f}%). The typical "
           f"self-tender premium in the literature is low-teens — so the trade works, "
           f"*narrowly*, and **{R['be_share_above']:.0%}** of events needed more than the "
           f"{R['premium']:.0f}% we assumed. Premium-free is not assumption-free: this number "
           f"still depends on where we anchor the 'pre-announcement' price (9.18% at 2 "
           f"sessions back, {R['anchor21_be']:.2f}% at 21) and on the frictions we charge — "
           f"buying 99 shares of a micro-cap at a 100 bps touch pushes it to "
           f"**{R['c100_be']:.2f}%**, i.e. all the way to the premium we assumed, and the "
           f"round trip down to {R['c100']:+.2f}% (*t* = {R['c100_t']:+.2f})."),
        md("## 6. And the entry price has been getting worse\n\n"
           f"Split at 2018 — a date fixed in advance, not searched. Two point estimates are "
           f"not a contrast, so we test the *difference* itself (Welch *t*, late minus "
           f"early). The announcement pop has grown from **{R['era_e_runup']:.2f}%** to "
           f"**{R['era_l_runup']:.2f}%** (*t* = {R['d_runup_t']:+.2f}) and the breakeven "
           f"premium with it, **{R['era_e_be']:.2f}% → {R['era_l_be']:.2f}%** "
           f"(*t* = {R['d_be_t']:+.2f}) — both real changes, and the second lands within a "
           f"point of the premium we assumed.\n\n"
           f"The profit fell too, from **{R['era_e_hedged']:+.2f}%** (*t* = "
           f"{R['era_e_t']:+.2f}) to **{R['era_l_hedged']:+.2f}%** (*t* = "
           f"{R['era_l_t']:+.2f}) — but that *decline* is only *t* = {R['d_hedged_t']:+.2f}, "
           f"so we say it carefully: what has demonstrably deteriorated is the price you pay "
           f"to get in, and the modern round trip no longer clears *t* = 2 on its own."),
        md("## 7. The killer: the rule that gives you the edge also caps it\n\n"
           f"Ninety-nine shares. That is the whole allowance. At the default assumptions the "
           f"trade is **${R['dollar_event']:.0f} per event** and about "
           f"**${R['cap5000_yr']:,.0f} a year** before tax, per account — and only "
           f"**${R['cap1000_yr']:,.0f} a year** on a $10 stock, where the flat broker fee "
           f"alone is 300 bps and the *t* falls to {R['cap1000_t']:+.2f}. The version with "
           f"real money in it is the round-lot version — which is exactly the version "
           f"proration hits.\n\n"
           "> 🔬 *For the quants:* scaling it is not a capital constraint but a legal one. "
           "The 'if we could do it at size' calendar-time sleeve is priced in notebook 02 "
           "and loses the excess-of-cash Sharpe race to SPY anyway, at 66% volatility."),
        md("## 8. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "Plant a run-up and a post-expiry give-back into a synthetic panel: the estimator "
           "must find both. Plant nothing: it must find nothing. This is **synthetic data**, "
           "not the tender list — it proves the missing give-back on the real tape is a fact "
           "about tenders, not a broken harness."),
        code(SYN_CELL),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed**, and the split is between what the tape says and what we "
           f"assumed. **The tape is emphatic about the bad news:** the pop takes "
           f"**{R['runup']:.2f}%** before you can buy (*t* = {R['runup_abn_t']:+.2f}) and the "
           f"post-expiry give-back priority feeds on is absent "
           f"(*t* = {R['give_t']:+.2f}). **Every good number is assumption-carried:** the "
           f"headline **{R['hedged']:+.2f}%** flips sign at a 9% premium, and even the "
           f"'value of priority' **{R['priority']:+.2f}%** flips at "
           f"{R['prio_zero']:.1f}%. Without a premium the tape says only this: you would need "
           f"a **{R['be_mean']:.2f}%** clearing premium to break even, "
           f"**{R['era_l_be']:.2f}%** on modern deals, **{R['c100_be']:.2f}%** if the stock "
           f"is a micro-cap.\n"
           f"- **Tradability — Mirage.** Ninety-nine shares, ~${R['cap5000_yr']:,.0f} a year "
           f"at best, one flat broker fee at a time. Small, capped by law, and impossible "
           f"to scale by construction."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 928 — Odd-Lot Priority — the teardown\n\n"
           "The tape legs and their HAC *t*s, the two round-trip identities, the "
           "premium-free breakeven premium and what it still assumes, the sweep over every "
           "declared proxy, the era cut with the *contrast* tested, the calendar-time "
           "excess-of-cash race and the live synthetic control. "
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`), "
           "%s → %s, %d usable event windows, as-of 2026-06-30."
           % (R["fp"], R["start"], R["end"], R["n_events"])),
        code("R = %r" % (R,)),
        md("## The estimand, written out\n\n"
           "With `p_pre` the close 6 sessions before commencement, `p_entry` the close at "
           "*t+1* (the one execution lag), `p_clear = p_pre × (1 + premium)` the assumed "
           "clearing price, `p_post` the close 5 sessions after expiry and `f` the "
           "round-lot proration fill:\n\n"
           "```\n"
           "1 + r_odd          = (1 + premium) / (1 + runup)\n"
           "r_odd - r_round    = (1 - f) x [(p_clear - p_post) / p_entry + cost]\n"
           "breakeven premium  = (p_entry / p_pre) x (1 + r_SPY + frictions) - 1\n"
           "```\n\n"
           "Both identities are asserted in `tests/test_strategy.py`, not just claimed. "
           "**Read line 2 carefully:** `p_clear` is the assumed price, so 'what priority is "
           "worth' carries the premium assumption exactly as line 1 does — it is not the "
           "measured half of this study, and the premium sweep below shows it changing "
           "sign. Line 3 is the only one free of the premium.\n\n"
           "> 💡 *In plain words:* the premium is a number we assume, the run-up and the "
           "post-expiry price are numbers the market printed. The third line asks the tape "
           "how big the assumed number would have to be for the trade to be worth doing."),
        md("## The tape legs (no premium assumption in the first four)"),
        code(
            "rows = [('run-up t-6 -> t+1', R['runup'], R['runup_t']),\n"
            "        ('run-up, abnormal', R['runup_abn'], R['runup_abn_t']),\n"
            "        ('offer-window drift', R['drift'], R['drift_t']),\n"
            "        ('post-expiry give-back', R['give'], R['give_t']),\n"
            "        ('permanent reprice, abnormal', R['perm_abn'], R['perm_abn_t'])]\n"
            "for label, mean, t in rows:\n"
            "    flag = '' if abs(t) >= 2 else '   <- not significant'\n"
            "    print('%-28s mean %+6.2f%%   HAC t %+6.2f%s' % (label, mean, t, flag))\n"
            "print('\\nhit rate on the run-up %.2f  |  on the give-back %.2f (a coin flip)'\n"
            "      % (R['runup_hit'], R['give_hit']))"
        ),
        md("The give-back is the load-bearing leg and it is **absent**. Odd-lot priority is "
           "the right to be paid the tender price instead of holding the stub — worth "
           "something only if the stub is worth less. The stub is not worth less: the stock "
           f"keeps **{R['perm_abn']:+.2f}%** abnormal permanently, and in "
           f"**{R['share_post_above_clear']:.0%}** of events `p_post > p_clear`."),
        md("## The priced round trips (all four carry the premium) and the premium-free "
           "breakeven"),
        code(
            "print('ALL FOUR of these carry the assumed %.0f%% clearing premium:' % R['premium'])\n"
            "print('odd lot, net                : %+6.2f%%  HAC t %+5.2f' % (R['odd'], R['odd_t']))\n"
            "print('odd lot, net + SPY-hedged   : %+6.2f%%  HAC t %+5.2f  boot CI [%+.2f, %+.2f]'\n"
            "      % (R['hedged'], R['hedged_t'], R['hedged_lo'], R['hedged_hi']))\n"
            "print('round lot, net (f=0.35)     : %+6.2f%%  HAC t %+5.2f' % (R['round_lot'], R['round_t']))\n"
            "print('value of odd-lot priority   : %+6.2f%%  HAC t %+5.2f   <- assumption-carried too'\n"
            "      % (R['priority'], R['priority_t']))\n"
            "print('\\nthe one number free of the premium:')\n"
            "print('breakeven premium: mean %.2f%%  median %.2f%%  p75 %.2f%%  '\n"
            "      '(%.0f%% of events need more than the assumed %.0f%%)'\n"
            "      % (R['be_mean'], R['be_med'], R['be_p75'], R['be_share_above']*100, R['premium']))\n"
            "print('  ...but it still inherits the anchor (%.2f%% at 21 sessions back), the'\n"
            "      % R['anchor21_be'])\n"
            "print('  assumed offer length and the frictions (%.2f%% at a 100 bps touch).'\n"
            "      % R['c100_be'])"
        ),
        md("## Premium sweep — the assumption that decides the sign of **both** arms\n\n"
           "The *t* = +4.3 headline is bought with the premium assumption. So is the "
           "*t* = +4.33 attached to 'the value of priority': it is the same assumption "
           "wearing a different label, and it crosses zero at "
           f"{R['prio_zero']:.1f}%. The tape only pins the breakeven."),
        code(
            "print('assumed   hedged round trip        value of priority')\n"
            "for p, m, t, pv, pvt in [(5, R['prem5'], R['prem5_t'], R['prio5'], R['prio5_t']),\n"
            "                         (9, R['prem9'], R['prem9_t'], R['prio9'], R['prio9_t']),\n"
            "                         (13, R['hedged'], R['hedged_t'], R['priority'], R['priority_t']),\n"
            "                         (17, R['prem17'], R['prem17_t'], R['prio17'], R['prio17_t']),\n"
            "                         (25, R['prem25'], R['prem25_t'], R['prio25'], R['prio25_t'])]:\n"
            "    mark = '  <- default' if p == 13 else ''\n"
            "    print('%2d%%       %+7.2f%% (t %+6.2f)      %+6.2f%% (t %+6.2f)%s'\n"
            "          % (p, m, t, pv, pvt, mark))\n"
            "print('\\nhedged round trip crosses zero between 9%% and 13%%;'\n"
            "      ' value of priority crosses at %.1f%%.' % R['prio_zero'])\n"
            "print('at the breakeven premium (%.2f%%) priority is worth only %+.2f%% (t %+.2f).'\n"
            "      % (R['be_mean'], R['prio_at_be'], R['prio_at_be_t']))"
        ),
        md("## Proration, cost, capacity, borrow — the rest of the assumption surface"),
        code(
            "print('proration sweep (value of priority) - sets the SIZE; the premium sets the SIGN:')\n"
            "for f, v in [(0.15, R['pro15']), (0.35, R['pro35']), (0.75, R['pro75']), (1.00, R['pro100'])]:\n"
            "    print('   fill %.2f -> %+5.2f%%%s' % (f, v, '   <- under-subscribed: worth nothing' if f == 1.0 else ''))\n"
            "print('   (HAC t is invariant at %+.2f across this whole row - it scales in (1-f))'\n"
            "      % R['priority_t'])\n"
            "print('\\nflat broker fee on a $%.0f lot: $0 -> %+5.2f%%   $50 -> %+5.2f%% (t=%+.2f)'\n"
            "      % (R['position'], R['fee0'], R['fee50'], R['fee50_t']))\n"
            "print('bps touch (charged 3x: buy + SPY hedge round trip), $30 fee:')\n"
            "print('   15 bps -> %+5.2f%% (t=%+.2f)   100 bps -> %+5.2f%% (t=%+.2f)  '\n"
            "      '<- micro-cap touch kills it, breakeven %.2f%%'\n"
            "      % (R['hedged'], R['hedged_t'], R['c100'], R['c100_t'], R['c100_be']))\n"
            "print('borrow on the short SPY leg  : 300 bps -> %+5.2f%% (t=%+.2f)  [not the binding cost]'\n"
            "      % (R['borrow300'], R['borrow300_t']))\n"
            "print('offer length 40 td           : %+5.2f%% (t=%+.2f)' % (R['exp40'], R['exp40_t']))\n"
            "print('anchor 21 sessions back      : %+5.2f%% (t=%+.2f), breakeven %.2f%%  <- the most fragile choice'\n"
            "      % (R['anchor21'], R['anchor21_t'], R['anchor21_be']))\n"
            "print('\\ncapacity (a 99-share lot is 99 x the share price):')\n"
            "for pos, m, t, yr in [(1000, R['cap1000'], R['cap1000_t'], R['cap1000_yr']),\n"
            "                      (5000, R['hedged'], R['hedged_t'], R['cap5000_yr']),\n"
            "                      (9900, 3.82, 4.68, R['cap9900_yr'])]:\n"
            "    print('   $%5d lot -> %+5.2f%% (t=%+.2f)  ~$%s per year' % (pos, m, t, format(yr, ',')))"
        ),
        md("## Era cut (split 2018-01-01, fixed in advance) — and the contrast, tested\n\n"
           "Two point estimates are not a contrast. The desk rule is that a sub-period claim "
           "is a claim about a *difference*, so the difference gets its own test (Welch *t* "
           "on the per-event vectors, late minus early)."),
        code(
            "print('2010-2017 (n=%d): run-up %+5.2f%%  hedged %+5.2f%% (t=%+.2f)  breakeven %5.2f%%'\n"
            "      % (R['era_e_n'], R['era_e_runup'], R['era_e_hedged'], R['era_e_t'], R['era_e_be']))\n"
            "print('2018-2025 (n=%d): run-up %+5.2f%%  hedged %+5.2f%% (t=%+.2f)  breakeven %5.2f%%   <- t < 2'\n"
            "      % (R['era_l_n'], R['era_l_runup'], R['era_l_hedged'], R['era_l_t'], R['era_l_be']))\n"
            "print('\\nthe CHANGE itself (Welch t, late - early):')\n"
            "for lbl, gap, t in [('run-up', R['d_runup'], R['d_runup_t']),\n"
            "                    ('breakeven premium', R['d_be'], R['d_be_t']),\n"
            "                    ('hedged round trip', R['d_hedged'], R['d_hedged_t']),\n"
            "                    ('give-back', R['d_give'], R['d_give_t']),\n"
            "                    ('value of priority', R['d_priority'], R['d_priority_t'])]:\n"
            "    flag = 'real change' if abs(t) >= 2 else 'NOT a significant change'\n"
            "    print('   %-18s gap %+6.2f pp   Welch t %+5.2f   %s' % (lbl, gap, t, flag))\n"
            "print('\\n=> what demonstrably changed is the ENTRY PRICE, not the profit:')\n"
            "print('   the decay of the round trip is itself only t = %+.2f.' % R['d_hedged_t'])"
        ),
        md("## Calendar-time sleeve, excess-of-cash — the un-achievable scaled version\n\n"
           "Equal-weight across live positions, BIL when idle, both arms minus BIL. This "
           "applies the odd-lot **fill** assumption at portfolio scale — which is exactly "
           "what the odd-lot rule forbids — *and* it carries the assumed 13% clearing "
           "premium, so its positive mean is not evidence for anything. It is priced only "
           "to close the door.\n\n"
           "> 💡 *In plain words:* even if the law let you do this trade with real money, it "
           "would not have been a better use of the money than owning the index."),
        code(
            "print('%s -> %s, %d sessions, invested %.1f%% of days, %.2f live names, vol %.1f%%'\n"
            "      % (R['cal_start'], R['cal_end'], R['cal_days'], R['cal_invested'],\n"
            "         R['cal_live'], R['cal_vol']))\n"
            "print('sleeve excess Sharpe  gross %+.3f  net %+.3f' % (R['sh_gross'], R['sh_net']))\n"
            "print('SPY    excess Sharpe        %+.3f      -> advantage %+.3f (HAC t on the daily diff %+.2f)'\n"
            "      % (R['sh_spy'], R['sh_adv'], R['sh_t']))\n"
            "print('higher mean, lower Sharpe: 1.6 single names at full weight is a 66%-vol sleeve.')"
        ),
        md("## Live synthetic control — the machinery is unbiased\n\n"
           "**Synthetic data**, not the tender list. Planted: a +500 bps run-up and a "
           "−400 bps post-expiry give-back, both of which must be recovered. Null: nothing "
           "planted, nothing may be found."),
        code(SYN_CELL),
        code(
            "import numpy as np\n"
            "nulls = [st.synthetic_detect(*data.synthetic_panel(n_events=60,\n"
            "                                                   signal_strength=0.0,\n"
            "                                                   seed=928 + s)[:2])\n"
            "         for s in range(6)]\n"
            "ru = np.array([n['runup_bps'] for n in nulls])\n"
            "gb = np.array([n['giveback_bps'] for n in nulls])\n"
            "print('null x6: run-up mean %+.0f bps (sd %.0f), |t|>=2 in %d/6'\n"
            "      % (ru.mean(), ru.std(ddof=1), sum(abs(n['runup_t']) >= 2 for n in nulls)))\n"
            "print('null x6: give-back mean %+.0f bps, |t|>=2 in %d/6'\n"
            "      % (gb.mean(), sum(abs(n['giveback_t']) >= 2 for n in nulls)))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed**, split by leg: **Real on the announcement run-up · None on "
           f"the give-back the claim depends on.** The run-up is "
           f"**{R['runup']:+.2f}%** (abnormal {R['runup_abn']:+.2f}%, HAC *t* = "
           f"{R['runup_abn_t']:+.2f}) with no proxy anywhere in it; the post-expiry give-back "
           f"is **{R['give']:+.2f}%**, *t* = {R['give_t']:+.2f}, median 0.00%. **Every** "
           f"profitable number in the study is the assumed {R['premium']:.0f}% clearing "
           f"premium meeting the tape — the hedged round trip **{R['hedged']:+.2f}%** "
           f"(*t* = {R['hedged_t']:+.2f}) *and* the 'value of priority' "
           f"**{R['priority']:+.2f}%** (*t* = {R['priority_t']:+.2f}) alike; they cross zero "
           f"at 9-13% and {R['prio_zero']:.1f}% respectively. Premium-free, the tape says the "
           f"breakeven is **{R['be_mean']:.2f}%**, **{R['era_l_be']:.2f}%** post-2018 (the "
           f"rise is itself significant, Welch *t* = {R['d_be_t']:+.2f}) and "
           f"**{R['c100_be']:.2f}%** at a micro-cap 100 bps touch. The synthetic control "
           f"recovers a planted run-up ({R['syn_pl_runup']:+d} bps of {500}) and a planted "
           f"give-back ({R['syn_pl_give']:+d} bps of −400) and is silent on the null "
           f"({R['syn_nl_runup']:+d} bps, *t* = {R['syn_nl_runup_t']:+.2f}) — those are "
           f"`docs/results.md`'s 150-event panels; the live cells above run a 90-event one "
           f"and land in the same place — so the missing give-back is a fact about tenders, "
           f"not a broken harness.\n"
           f"- **Tradability — Mirage.** The entitlement caps the position at 99 shares: "
           f"**${R['dollar_event']:.0f} per event**, **~${R['cap5000_yr']:,.0f} a year** at a "
           f"$50 share price and **${R['cap1000_yr']:,.0f}** at a $10 one, where the flat fee "
           f"is 300 bps and *t* = {R['cap1000_t']:+.2f}. The scaled fantasy sleeve loses the "
           f"excess-of-cash Sharpe race to SPY ({R['sh_net']:+.3f} vs {R['sh_spy']:+.3f}) at "
           f"{R['cal_vol']:.0f}% vol. Nothing bankable at desk size."),
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
