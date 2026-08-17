"""Generate the two narrative notebooks for Study 919 (Methodology Shock).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the
fast synthetic control, and they are never presented under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. QQQ/SPY, SPY/IWM, IWM/MDY total
# return, market-model CAR, tradable window [+1,+10], 1993-01-29 -> 2026-06-30.
R = dict(
    start="1993-01-29", end="2026-06-30", n_days=8411, fp="547814cd71c7",
    n_events_listed=8, n_events=7,
    ann_car=-14.1, ann_med=-22.8, ann_sd=122.9, ann_t=-0.30, ann_hac=-0.30,
    ann_hit=43, ann_p=0.802, ann_null_mu=-3.4, ann_null_sd=64.2,
    ann_null_lo=-129, ann_null_hi=124, ann_mde=114,
    ann_boot_lo=-93, ann_boot_hi=73, ann_blk_lo=-108, ann_blk_hi=79,
    eff_car=35.7, eff_med=36.9, eff_sd=81.6, eff_t=1.16, eff_hac=0.86,
    eff_hit=57, eff_p=0.541, eff_mde=75,
    eff_boot_lo=-15, eff_boot_hi=96, eff_blk_lo=-42, eff_blk_hi=114,
    pre_car=-33.2, pre_t=-3.43, pre_p=0.451,
    worst_p=0.423, bonf=1.000, n_windows=9,
    jk_min=-48.5, jk_min_t=-1.32, jk_max=9.7,
    era_e_n=3, era_e_car=-60.0, era_e_t=-1.23,
    era_l_n=4, era_l_car=20.4, era_l_t=0.28,
    trade_gross=-7.7, trade_net=-29.3, trade_t=-0.62, trade_win=43,
    naive_gross=46.2, naive_net=24.3, naive_t=0.51,
    eff_trade_gross=43.2, eff_trade_net=21.8, eff_trade_t=0.70,
    cost0_net=-8.2, cost25_net=-115.5, cost25_t=-2.30,
    fin_rate=200, fin_charge=0.52, fin0_net=-28.8, fin500_net=-30.1,
    dc_events=5, dc_dropped=2,
    live_days=50, total_days=4802, overlay_total=65, overlay_sharpe=0.187,
    overlay_t=0.37, spy_sharpe=0.542,
    fix_old_date="2011-03-24", fix_new_date="2011-04-05",
    fix_old_car=-120.1, fix_new_car=36.3, fix_old_pooled=-36.4,
    syn_planted=250, syn_rec=314.9, syn_rec_t=3.21, syn_rec_p=0.000, syn_rec_hit=92,
    syn_null=64.9, syn_null_p=0.270, syn_seed_mu=17.4, syn_seed_sd=53.5,
)


HEADER = f"""# Study 919 — Methodology Shock 📐

**When an index changes its own rules, can you front-run the trade every tracker must make?**

An index is a rulebook. Every fund that tracks it is contractually obliged to hold whatever
the rulebook says — so when the *rules themselves* change (a special rebalance to cap
concentration, a switch to float-adjusted weights, a new eligibility filter, a change to how
constituents migrate), trillions of dollars of passive money must trade a large basket on a
publicly announced date. The folklore says: buy the affected wrapper against an unaffected
sibling between the announcement and the effective date.

We test it on **{R['n_events_listed']} hardcoded index rule changes** — Nasdaq-100 special
rebalances, the S&P float-adjustment phases, the S&P multiple-share-class ban and its
reversal, Russell banding and the move to semi-annual reconstitution — with
**QQQ vs SPY**, **SPY vs IWM** and **IWM vs MDY** as treated/sibling pairs, on daily
total-return closes, {R['start']} → {R['end']} ({R['n_days']:,} sessions).

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
live cells run the fast offline synthetic control and are labelled as such. As-of
{R['end']}.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Why anyone would expect this to work\n\n"
           "The mechanism is not folklore — it is a contract. If S&P announces that the "
           "index will be reweighted on a Friday, every S&P tracker on earth **must** trade "
           "that reweighting, and everybody knows it weeks in advance. Forced buyers on a "
           "known date is about as clean a set-up as markets offer.\n\n"
           "The catch is *where* the forced trade lands. A special rebalance that halves "
           "one company's weight is enormous **for that company** — and gets diluted across "
           "the other four hundred names in the wrapper you can actually buy. This study "
           "asks only the version a price-taker can act on: does the shock survive at the "
           "**wrapper** level, after you hedge out the market and pay the spread?\n\n"
           "> 🔬 *For the quants:* the constituent-level demand-curve literature (Shleifer "
           "1986, Harris & Gurel 1986, Greenwood 2005) is strong and we do not dispute it. "
           "The wrapper-level residual is a different, much smaller quantity."),
        md("## 2. How we measure a 'shock'\n\n"
           "For each rule change we take the affected fund and a sibling the change does "
           "*not* touch — QQQ against SPY for Nasdaq rules, SPY against IWM for S&P rules, "
           "IWM against MDY for Russell rules. We learn how the two normally move together "
           "over the year *before* the announcement, then ask how much the affected fund "
           "moved **beyond** what the sibling explains, over the ten sessions after the "
           "news. That excess is the shock.\n\n"
           "One rule we never bend: the announcement is public at the closing bell of day "
           "zero, so our trade starts the *next* day. We never earn the return of the day "
           "the news broke."),
        code(
            "R = dict(ann_car=%r, ann_p=%r, eff_car=%r, eff_p=%r,\n"
            "         ann_hit=%r, eff_hit=%r, n_events=%r)\n"
            "print('announcement day  : average excess move %%+.1f bps   (chance of seeing this by luck: %%.0f%%%%)'\n"
            "      %% (R['ann_car'], R['ann_p']*100))\n"
            "print('effective day     : average excess move %%+.1f bps   (chance of seeing this by luck: %%.0f%%%%)'\n"
            "      %% (R['eff_car'], R['eff_p']*100))\n"
            "print('the two legs point in OPPOSITE directions, on %%d events each' %% R['n_events'])"
            % (R["ann_car"], R["ann_p"], R["eff_car"], R["eff_p"],
               R["ann_hit"], R["eff_hit"], R["n_events"])
        ),
        md(f"## 3. The answer: nothing there\n\n"
           f"The announcement leg moves the affected wrapper **{R['ann_car']:+.1f} bps** "
           f"relative to its sibling. The effective leg moves it **{R['eff_car']:+.1f} bps** "
           f"— the *other way*. Neither is remotely unusual: we re-ran the same measurement "
           f"on two thousand sets of **randomly chosen** dates, and swings this size show "
           f"up by pure chance most of the time "
           f"(*p* = {R['ann_p']:.2f} and {R['eff_p']:.2f}).\n\n"
           f"We tried {R['n_windows']} different windows — one day, three, five, ten, "
           f"twenty-one, and several straddling the announcement. The best of them still "
           f"had a {R['worst_p']:.0%} chance of being luck.\n\n"
           f"> ⚠️ *How thin this is:* one of the eight announcement dates was originally "
           f"transcribed wrong ({R['fix_old_date']} instead of {R['fix_new_date']} for the "
           f"2011 Nasdaq-100 rebalance). Correcting it flipped that event from "
           f"{R['fix_old_car']:+.0f} to {R['fix_new_car']:+.0f} bps and moved the pooled "
           f"headline from {R['fix_old_pooled']:+.1f} to {R['ann_car']:+.1f} bps. When one "
           f"typo can move your answer by 22 bps, you do not have a result."),
        md(f"## 4. The trap this study is really about\n\n"
           f"One window looked spectacular. Over the five days **before** the announcement, "
           f"the affected wrapper lagged by {R['pre_car']:.1f} bps with a *t*-statistic of "
           f"**{R['pre_t']:.2f}** — the kind of number that gets a chart into a pitch deck.\n\n"
           f"It is an illusion. With only seven events, the *t*-statistic has to guess how "
           f"variable these numbers are from seven data points, and those seven happened to "
           f"land close together. The random-date test, which knows the true variability, "
           f"prices exactly the same {R['pre_car']:.0f} bps at *p* = {R['pre_p']:.2f} — "
           f"a coin flip.\n\n"
           f"> 🔬 *For the quants:* the cross-event *t* estimates its denominator from n = 7; "
           f"the randomisation null for a 5-day CAR on this tape has a standard deviation "
           f"several times the realised cross-event dispersion. Nine windows were examined; "
           f"after Bonferroni every adjusted *p* is {R['bonf']:.3f}."),
        md(f"## 5. And the trade itself does not pay\n\n"
           f"Take the trade anyway: buy the affected wrapper, short the right amount of the "
           f"sibling, hold ten days. Before any cost it returns **{R['trade_gross']:+.1f} bps** "
           f"per event. After 5 bps of spread on each leg in and out, plus borrow on the "
           f"short and financing on the bit that is not self-funding: "
           f"**{R['trade_net']:+.1f} bps**, winning {R['trade_win']}% of the time. Every "
           f"single cost/borrow combination we tried is negative, including the one where "
           f"trading is free ({R['cost0_net']:+.1f} bps).\n\n"
           f"An investor who parked in T-bills and put this on at the "
           f"{R['dc_events']} events that fall inside the T-bill era earned "
           f"**{R['overlay_total']:+d} bps** over nineteen years — {R['live_days']} live "
           f"days out of {R['total_days']:,}, an excess-of-cash Sharpe of "
           f"**{R['overlay_sharpe']:+.2f}** against SPY's {R['spy_sharpe']:+.2f}. That is "
           f"about 3 bps a year. And it is only positive because the T-bill era happens to "
           f"start after the two events that lost the most."),
        md("## 6. Live check — the machinery does work (offline synthetic)\n\n"
           "**This cell is synthetic, not the real tape.** We build a fake world with a "
           "genuine, planted shock after each announcement, and check the detector finds it; "
           "then a matched world with nothing planted, and check it stays quiet. If the "
           "detector passes both, the flat real-tape answer is about the market, not a bug."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from methodology_shock import data, strategy as st\n"
            "planted = st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, seed=919)[:2],\n"
            "                              window=(1, 10), n_draws=300)\n"
            "null    = st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=919)[:2],\n"
            "                              window=(1, 10), n_draws=300)\n"
            "print('SYNTHETIC (not the real tape)')\n"
            "print('  planted a 250 bps shock -> detector reports %+.0f bps, p = %.3f  (must fire)'\n"
            "      % (planted['mean_car_bps'], planted['placebo_p']))\n"
            "print('  planted nothing         -> detector reports %+.0f bps, p = %.3f  (must not fire)'\n"
            "      % (null['mean_car_bps'], null['placebo_p']))"
        ),
        md(f"## 7. What we can and cannot claim\n\n"
           f"Honesty about power: with seven events, the smallest shock this design could "
           f"ever have called real is about **{R['ann_mde']} bps**. A genuine 20-40 bps "
           f"footprint — which is roughly what the mechanism should produce once a "
           f"basket-wide reweighting is spread across hundreds of names — would have been "
           f"invisible to us. So the finding is not 'index rule changes do nothing'. It is "
           f"the narrower, tradable statement: **nothing large enough to pay for the spread "
           f"shows up in the wrapper you can buy.**"),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Announcement leg {R['ann_car']:+.1f} bps "
           f"(*p* = {R['ann_p']:.3f}), effective leg {R['eff_car']:+.1f} bps "
           f"(*p* = {R['eff_p']:.3f}) — opposite signs, both indistinguishable from randomly "
           f"chosen dates. All {R['n_windows']} windows have a Bonferroni-adjusted "
           f"*p* of {R['bonf']:.3f}; both bootstrap CIs straddle zero; the two eras disagree.\n"
           f"- **Tradability — Mirage.** The hedged trade is negative **before** costs "
           f"({R['trade_gross']:+.1f} bps) and worse after ({R['trade_net']:+.1f} bps). Every "
           f"cell of the cost sweep is negative. The only version that looks profitable is "
           f"the unhedged one, and its {R['naive_net']:+.1f} bps is market beta wearing a "
           f"costume."),
        md(f"---\n\n*Every real-tape number above is frozen from "
           f"[`docs/results.md`](../docs/results.md) (Fingerprint `{R['fp']}`, as-of "
           f"{R['end']}). The only cell that computes anything live is the synthetic "
           f"control, and it is labelled.*"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 919 — Methodology Shock — the teardown\n\n"
           "A market-model event study over a hardcoded calendar of index **rule changes**, "
           "priced against a randomisation (placebo) null because the treated sample has "
           "seven observations. Inside: the two legs, the nine-window sweep with Bonferroni, "
           "the drop-one jackknife, the era cut, the beta-hedged costed pair with a borrow "
           "sweep, the deployed-capital race, and the live synthetic control.\n\n"
           "Design, in one line: fit `r_treated = a + b * r_control` on the 250 sessions "
           "ending 21 sessions before the event, cumulate `r_treated - a - b * r_control` "
           "over the window, pool across events. One execution lag — the announcement is "
           "public at the day-0 close, the tradable window starts at **+1**.\n\n"
           "> 💡 *In plain words:* we ask how much the affected fund moved beyond what its "
           "sibling explains, and then check whether random dates do the same thing just as "
           "often.\n\n"
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`)."
           % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The headline — pooled CAR on both legs, window [+1,+10]\n\n"
           "Beta-adjusted CARs in bps, seven usable events per leg. The placebo *p* comes "
           "from 2,000 draws that replace each real date with a random session on the same "
           "pair (excluding ±30 sessions around any real event) and recompute identically.\n\n"
           "Two caveats stated before the numbers, not after. (a) **The legs are not "
           "independent**: both multiple-share-class rows took effect on the day they were "
           "announced, so 2 of the 7 observations are shared. They are shown side by side "
           "as a sign check and never pooled. (b) **One announcement date in the calendar "
           "was wrong** — the 2011 Nasdaq-100 rebalance was dated 2011-03-24 and stamped "
           "`exact`; the press release is 2011-04-05. Fixing it moved that event from "
           "−120.1 to +36.3 bps and the pooled announcement leg from −36.4 to −14.1 bps. "
           "A seven-observation study is exactly that sensitive to its own data entry.\n\n"
           "> 💡 *In plain words:* two legs, opposite signs, and both look like random dates."),
        code(
            "print(f\"announce  : n={R['n_events']}  CAR {R['ann_car']:+.1f} bps  median {R['ann_med']:+.1f}  \"\n"
            "      f\"sd {R['ann_sd']:.1f}  hit {R['ann_hit']}%\")\n"
            "print(f\"            cross-event t {R['ann_t']:+.2f}  HAC t (daily AR) {R['ann_hac']:+.2f}  \"\n"
            "      f\"placebo p {R['ann_p']:.3f}\")\n"
            "print(f\"            placebo null {R['ann_null_mu']:+.1f} +/- {R['ann_null_sd']:.1f} bps, \"\n"
            "      f\"95% band [{R['ann_null_lo']:+d}, {R['ann_null_hi']:+d}]\")\n"
            "print(f\"effective : n={R['n_events']}  CAR {R['eff_car']:+.1f} bps  median {R['eff_med']:+.1f}  \"\n"
            "      f\"sd {R['eff_sd']:.1f}  hit {R['eff_hit']}%\")\n"
            "print(f\"            cross-event t {R['eff_t']:+.2f}  HAC t (daily AR) {R['eff_hac']:+.2f}  \"\n"
            "      f\"placebo p {R['eff_p']:.3f}\")\n"
            "print(f\"\\nevent bootstrap CI  announce [{R['ann_boot_lo']:+d}, {R['ann_boot_hi']:+d}]  \"\n"
            "      f\"effective [{R['eff_boot_lo']:+d}, {R['eff_boot_hi']:+d}]  (bps)\")\n"
            "print(f\"block bootstrap CI  announce [{R['ann_blk_lo']:+d}, {R['ann_blk_hi']:+d}]  \"\n"
            "      f\"effective [{R['eff_blk_lo']:+d}, {R['eff_blk_hi']:+d}]  (bps)\")"
        ),
        md("## 2. Power, stated before the conclusion\n\n"
           "With n = 7 and a cross-event CAR dispersion of 123 bps, the smallest pooled CAR "
           "callable at 5% is ~114 bps; the placebo band agrees at ±~127 bps. This bounds "
           "what the null result means — it rules out a large wrapper-level footprint, not a "
           "20-40 bps one.\n\n"
           "> 💡 *In plain words:* a small effect could have been here and we would never "
           "have seen it. We can only say a big one is absent."),
        code(
            "print(f\"minimum detectable CAR (5%, n={R['n_events']}): announce {R['ann_mde']} bps, \"\n"
            "      f\"effective {R['eff_mde']} bps\")\n"
            "print(f\"placebo 95% band (announce leg): [{R['ann_null_lo']:+d}, {R['ann_null_hi']:+d}] bps\")"
        ),
        md("## 3. The window sweep — and the small-sample t-statistic trap\n\n"
           "Nine windows; the four that span day 0 or earlier are descriptive (not "
           "actionable) and labelled so. The `[-5,-1]` row is the instructive failure: a "
           "naive cross-event *t* of −3.43, and a randomisation *p* of 0.45.\n\n"
           "> 💡 *In plain words:* the *t*-test guesses how noisy the world is from seven "
           "numbers. The random-date test does not have to guess."),
        code(
            "print(f\"[-5,-1] descriptive: CAR {R['pre_car']:+.1f} bps  naive cross-event t {R['pre_t']:+.2f}  \"\n"
            "      f\"-> placebo p {R['pre_p']:.3f}\")\n"
            "print(f\"worst (smallest) placebo p across all {R['n_windows']} windows: {R['worst_p']:.3f}\")\n"
            "print(f\"Bonferroni-adjusted over {R['n_windows']} windows: {R['bonf']:.3f} for every one of them\")"
        ),
        md("## 4. Jackknife and era cut — how fragile a seven-point mean is\n\n"
           "Dropping one event at a time moves the pooled CAR anywhere between −48.5 and "
           "+9.7 bps — it changes **sign**. No sub-sample gets past |*t*| = 1.32. The era "
           "cut splits 3 against 4 observations and the halves disagree in sign; it is "
           "reported because the house style requires it, not because three points decide "
           "anything.\n\n"
           "> 💡 *In plain words:* remove one event and the answer changes shape. That is the "
           "definition of a result you should not trade."),
        code(
            "print(f\"jackknife: pooled CAR ranges {R['jk_min']:+.1f} .. {R['jk_max']:+.1f} bps \"\n"
            "      f\"(it changes sign); worst t {R['jk_min_t']:+.2f}\")\n"
            "print(f\"era pre-2015 (n={R['era_e_n']}): CAR {R['era_e_car']:+.1f} bps, t {R['era_e_t']:+.2f} \"\n"
            "      f\"<- n=3, decides nothing\")\n"
            "print(f\"era 2015->   (n={R['era_l_n']}): CAR {R['era_l_car']:+.1f} bps, t {R['era_l_t']:+.2f}  \"\n"
            "      f\"<- opposite sign\")"
        ),
        md("## 5. The tradable arm — beta-hedged, costed, borrow- and financing-swept\n\n"
           "Long 1 unit treated, short `beta` units of the sibling (beta from the same clean "
           "pre-event window), on at +1, off at +10. Cost `(1+beta) x 2 x cost_bps` per round "
           "trip, one-way x NAV; borrow on the short notional.\n\n"
           "**The beta-hedged pair is not dollar-neutral, and this study does not pretend it "
           "is.** Long 1 against short `beta` leaves `1 - beta` units of NAV as a real net "
           "position — net long +0.52 on the share-class-ban event (beta 0.48), net short "
           "−0.23 on NDX 2023 (beta 1.23). Only the naive 1x/1x variant finances itself "
           "exactly. That residual is charged (or credited) at an assumed bill rate, which "
           "is what makes the reported number an excess-of-cash return instead of one merely "
           "called that. It is small — mean +0.52 bps at 200 bps/yr, swept 0 to 500 — and it "
           "is charged anyway.\n\n"
           "The naive 1x/1x variant is shown because it is the trap: its positive number is "
           "unhedged market exposure (SPY's beta to IWM is 0.48-0.81), not a rule-change "
           "effect.\n\n"
           "> 💡 *In plain words:* hedge properly and the trade loses before you pay anyone."),
        code(
            "print(f\"beta-hedged, announce leg : gross {R['trade_gross']:+.1f} bps  \"\n"
            "      f\"net {R['trade_net']:+.1f} bps (t {R['trade_t']:+.2f})  win {R['trade_win']}%\")\n"
            "print(f\"naive 1x/1x variant       : gross {R['naive_gross']:+.1f} bps  \"\n"
            "      f\"net {R['naive_net']:+.1f} bps (t {R['naive_t']:+.2f})  <- unhedged beta\")\n"
            "print(f\"beta-hedged, effective leg: gross {R['eff_trade_gross']:+.1f} bps  \"\n"
            "      f\"net {R['eff_trade_net']:+.1f} bps (t {R['eff_trade_t']:+.2f})\")\n"
            "print(f\"\\nfinancing on the residual (1-beta) exposure at {R['fin_rate']} bps/yr: \"\n"
            "      f\"mean {R['fin_charge']:+.2f} bps  (net {R['fin0_net']:+.1f} at 0 bps/yr, \"\n"
            "      f\"{R['fin500_net']:+.1f} at 500)\")\n"
            "print(f\"every cell of the cost x borrow sweep is negative: from {R['cost0_net']:+.1f} bps \"\n"
            "      f\"at ZERO cost to {R['cost25_net']:+.1f} bps (t {R['cost25_t']:+.2f}) at the worst corner\")\n"
            "print('the only |t| >= 2 on the tradable arm says the trade reliably LOSES money')"
        ),
        md("## 6. Deployed-capital race (excess-of-cash)\n\n"
           "Park in BIL, overlay the pair only inside event windows. Sparse by construction — "
           "50 live days out of 4,802 — so the Sharpe is read as a scale marker, not a "
           "portfolio statistic.\n\n"
           "**BIL starts 2007-05-30, so 2 of the 7 announcement events fall outside the cash "
           "era and are DROPPED.** They are not mapped onto BIL's first session: an earlier "
           "build did exactly that, stacking two pre-2007 events' P&L into the first ten days "
           "of the cash series and inventing a −3.8% drawdown out of nothing. Both dropped "
           "events happen to be losers, which is why the overlay below reads positive while "
           "the full-sample per-event mean is −29.3 bps net. The sub-sample is the flattering "
           "one, and it still earns 3 bps a year.\n\n"
           "> 💡 *In plain words:* nineteen years, five trades, a rounding error."),
        code(
            "print(f\"{R['dc_events']} events inside the BIL era ({R['dc_dropped']} dropped: pre-BIL), \"\n"
            "      f\"{R['live_days']} live days of {R['total_days']:,}\")\n"
            "print(f\"total excess {R['overlay_total']:+d} bps over nineteen years\")\n"
            "print(f\"overlay excess-of-cash Sharpe {R['overlay_sharpe']:+.3f} (HAC t {R['overlay_t']:+.2f})\")\n"
            "print(f\"reference: SPY excess-of-cash Sharpe {R['spy_sharpe']:+.3f} over the same BIL window\")"
        ),
        md("## 7. Live synthetic control — the detector is powered and unbiased\n\n"
           "**Synthetic, not the real tape.** Three pairs, twelve announcements, a planted "
           "+250 bps abnormal drift bled in over the ten sessions after each; then a matched "
           "null. The planted world must clear the placebo; the null must not."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from methodology_shock import data, strategy as st\n"
            "pl = st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, seed=919)[:2],\n"
            "                         window=(1, 10), n_draws=400)\n"
            "print('SYNTHETIC (never supports the real-tape stamp)')\n"
            "print(f\"  planted 250 bps: CAR {pl['mean_car_bps']:+.1f} bps, t {pl['t_cross_event']:+.2f}, \"\n"
            "      f\"placebo p {pl['placebo_p']:.3f}, hit {pl['hit_rate']:.0%}\")\n"
            "nulls = np.array([\n"
            "    st.run_event_study(*data.synthetic_panel(signal_strength=0.0, seed=919 + 7 * k)[:2],\n"
            "                       window=(1, 10))['mean_car_bps']\n"
            "    for k in range(6)\n"
            "])\n"
            "print(f\"  null across 6 seeds: mean {nulls.mean():+.1f} bps, sd {nulls.std(ddof=1):.1f} bps \"\n"
            "      f\"-> centred on zero\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Pooled CAR {R['ann_car']:+.1f} bps on the announcement leg "
           f"(placebo *p* = {R['ann_p']:.3f}) and {R['eff_car']:+.1f} bps on the effective leg "
           f"(*p* = {R['eff_p']:.3f}) — **opposite signs**, both inside a null band of "
           f"[{R['ann_null_lo']:+d}, {R['ann_null_hi']:+d}] bps. All {R['n_windows']} windows "
           f"carry a Bonferroni-adjusted *p* of {R['bonf']:.3f} (smallest raw *p* "
           f"{R['worst_p']:.3f}); the event bootstrap "
           f"[{R['ann_boot_lo']:+d}, {R['ann_boot_hi']:+d}] and block bootstrap "
           f"[{R['ann_blk_lo']:+d}, {R['ann_blk_hi']:+d}] both straddle zero; the eras "
           f"disagree in sign; the drop-one jackknife swings the pooled CAR from "
           f"{R['jk_min']:+.1f} to {R['jk_max']:+.1f} bps, through zero. Nothing clears "
           f"|*t*| >= 2 on the real tape. The synthetic control recovers a planted 250 bps shock at "
           f"{R['syn_rec']:+.0f} bps (*p* = {R['syn_rec_p']:.3f}) and stays centred on the "
           f"null ({R['syn_seed_mu']:+.1f} bps, sd {R['syn_seed_sd']:.1f}, six seeds), so the "
           f"flat result is the tape's, not the harness's. **Survivorship: none to name** — "
           f"four continuously listed wrappers chosen ex ante by the index each tracks; the "
           f"selection risk lives in the hand-assembled event list, which the jackknife, the "
           f"window sweep and the placebo test all interrogate — and in which one date was "
           f"found wrong and corrected in audit, moving the headline by 22 bps on its own.\n"
           f"- **Tradability — Mirage.** {R['trade_gross']:+.1f} bps gross, "
           f"{R['trade_net']:+.1f} bps net per event (*t* {R['trade_t']:+.2f}), "
           f"{R['trade_win']}% win rate; every cell of the cost x borrow sweep negative, "
           f"including the free-trading corner ({R['cost0_net']:+.1f} bps). The "
           f"deployed-capital overlay's {R['overlay_total']:+d} bps is "
           f"{R['dc_events']} post-2007 trades of noise (Sharpe {R['overlay_sharpe']:+.2f}, "
           f"HAC *t* {R['overlay_t']:+.2f}, against SPY's {R['spy_sharpe']:+.2f}) on a "
           f"sub-sample that excludes the two biggest losers.\n"
           f"- **Power caveat, stated plainly.** Minimum detectable CAR ~{R['ann_mde']} bps. "
           f"This design rules out a *large* wrapper-level methodology shock, not a small "
           f"one. The tradable claim — nothing big enough to pay for the spread — is what "
           f"the tape supports."),
        md(f"---\n\n*Every real-tape number above is frozen from "
           f"[`docs/results.md`](../docs/results.md) (Fingerprint `{R['fp']}`, as-of "
           f"{R['end']}), reproducible with `python examples/verify.py`. The only live "
           f"cell is the synthetic control, and it is banner-labelled.*"),
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
