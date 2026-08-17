"""Generate the two narrative notebooks for Study 927 (Dutch Auction).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every **real-tape** number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md`` line for line; the only live
cells run the fast **synthetic** control, and they are always introduced as synthetic so no
synthetic output ever appears under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. 145 SC TO-I modified Dutch
# auction self-tenders, 109 issuers, 2010-06-18 -> 2025-11-21, abnormal = issuer - SPY,
# daily total-return closes, one execution lag, as-of 2026-06-30.
R = dict(
    n_edgar=180, n_events=145, n_issuers=109, n_tapes=128,
    start="2010-06-18", end="2025-11-21", fp="c39ea6c65ccb",
    pre5=1.44, pre5_med=0.88, pre5_t=3.20, pre5_hit=58,
    ar0=4.72, ar0_med=3.26, ar0_t=7.93, ar0_hit=83,
    win=0.12, win_med=0.61, win_t=0.17, win_hit=55,
    m1=-0.25, m1_med=-1.08, m1_t=-0.29, m1_hit=43,
    m3=1.69, m3_med=-0.28, m3_t=0.88, m3_hit=50,
    m6=2.76, m6_med=-2.36, m6_t=0.71, m6_hit=47,
    jk_ar0_lo=7.81, jk_ar0_hi=8.86, jk_win_lo=-0.04, jk_win_hi=0.61,
    jk_m6_lo=-0.12, jk_m6_hi=0.83,
    ci_ar0_lo=3.55, ci_ar0_hi=6.01, ci_win_lo=-1.45, ci_win_hi=1.44,
    ci_m6_lo=-4.55, ci_m6_hi=11.95, ci_win_neg=41.4, ci_m6_neg=29.2,
    pb_ar0_mean=0.01, pb_ar0_sd=0.29, pb_ar0_p=0.0005,
    pb_win_p=0.9105, pb_m6_p=0.3025,
    cal_g_bps=-1.232, cal_g_sharpe=-0.169, cal_g_t=-0.63, cal_g_cum=-54.16,
    cal_n_bps=-2.213, cal_n_sharpe=-0.302, cal_n_t=-1.14, cal_n_cum=-69.24,
    cal_rt=93.7, cal_overstate=1.55, cal_drag=0.926,
    cal_vol=18.40, cal_live=1847, cal_days=4046, cal_names=1.57,
    race_g=0.168, race_n=0.111, race_spy=0.809, race_adv=-0.698, race_t=-2.00,
    race_spy_t=3.60, race_n_vol=21.43, race_spy_vol=17.13,
    era_e_n=63, era_e_ar0=3.10, era_e_ar0_t=5.78, era_e_win=1.16, era_e_win_t=1.53,
    era_e_m6=-1.15, era_e_m6_t=-0.36,
    era_l_n=82, era_l_ar0=5.97, era_l_ar0_t=6.29, era_l_win=-0.69, era_l_win_t=-0.65,
    era_l_m6=5.76, era_l_m6_t=0.90,
    liq_n=60, liq_ar0=3.43, liq_ar0_t=4.88, liq_ar0_hit=75,
    liq_win=0.14, liq_win_t=0.13, liq_m6=0.78, liq_m6_t=0.24, liq_m6_med=-2.89,
    liq_cal_sharpe=-0.177, liq_cal_t=-0.59,
    cost0_win=0.12, cost0_sharpe=-0.176, cost10_win=-0.28, cost10_sharpe=-0.302,
    cost25_win=-0.88, cost25_sharpe=-0.489, cost50_win=-1.88, cost50_sharpe=-0.792,
    borrow0=-0.295, borrow300=-0.369,
    shift_m5=-0.32, shift_m1=0.64, shift_m1_t=2.51, shift_p1=0.43, shift_p1_t=1.47,
    shift_p5=0.00, shift_p5_t=0.03,
    exp15_win=0.64, exp15_m6=1.49, exp25_win=-0.38, exp25_m6=2.89,
    exp30_win=-0.16, exp30_m6=1.96,
    syn_jump=470, syn_rec=468, syn_rec_t=23.06, syn_drift=300, syn_drift_exp=607,
    syn_drift_rec=661,
    syn_drift_t=2.74, syn_null_ar0_max=2.33, syn_null_ar0_fire=2,
    syn_null_m6_max=2.86, syn_null_m6_fire=7, syn_null_n=20,
)


HEADER = f"""# Study 927 — Dutch Auction 🔨

**When a board pays a premium to buy back its own stock in a single fixed auction, does the
tape reward whoever buys in behind it?**

A *modified Dutch auction* self-tender is the loudest repurchase there is. The company posts
a price range, invites holders to name their price, and buys a large block at one clearing
price inside a twenty-business-day window — usually starting at a **premium** to the market.
The folklore reads it as insiders declaring the stock cheap, and expects the stock to keep
out-performing afterwards.

We test it on **{R['n_events']} self-tenders** across **{R['n_issuers']} issuers**,
{R['start']} → {R['end']}, harvested by a single EDGAR full-text-search query over form
**SC TO-I** (so every event is checkable by accession number, not remembered). Abnormal
return = issuer **minus SPY**, daily **total-return** closes, one execution lag.

*Real-tape numbers below are the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`); the live cells run the offline **synthetic** control and are labelled as such.
As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. What actually happens on the day\n\n"
           "Forget theory for a second. Here is what the tape does across "
           f"{R['n_events']} of these offers — the stock's move *minus* the S&P 500's, so a "
           "market rally cannot be mistaken for good news about the company."),
        code(
            "R = dict(ar0=%r, ar0_t=%r, ar0_hit=%r, win=%r, win_t=%r, m6=%r, m6_med=%r, m6_t=%r)\n"
            "print('Announcement day     : %%+.2f%%%% vs the market   (t = %%+.2f, positive in %%d%%%% of cases)'\n"
            "      %% (R['ar0'], R['ar0_t'], R['ar0_hit']))\n"
            "print('The tender window    : %%+.2f%%%% vs the market   (t = %%+.2f)'\n"
            "      %% (R['win'], R['win_t']))\n"
            "print('Six months after     : %%+.2f%%%% on average, but %%+.2f%%%% for the TYPICAL one (t = %%+.2f)'\n"
            "      %% (R['m6'], R['m6_med'], R['m6_t']))"
            % (R["ar0"], R["ar0_t"], R["ar0_hit"], R["win"], R["win_t"],
               R["m6"], R["m6_med"], R["m6_t"])
        ),
        md("## 2. The whole story is one day wide\n\n"
           f"On the day the tender documents hit EDGAR, the stock jumps **{R['ar0']:.2f}%** "
           f"more than the market — and it does so **{R['ar0_hit']}% of the time**. That is "
           "not a subtle statistical whisper; it is one of the cleanest event effects on this "
           "desk.\n\n"
           "And then it stops. Buy at the close of the *next* session — the first moment you "
           f"could actually act on the filing — and the tender window pays **{R['win']:+.2f}%** "
           f"against the market. Six months after the offer expires, the *average* is "
           f"**{R['m6']:+.2f}%** but the *typical* company is **{R['m6_med']:+.2f}%** — the "
           "average is being carried by a handful of names that went on to do well for "
           "reasons that have nothing to do with the buyback.\n\n"
           "> 🔬 **For the quants.** Announcement *t* = "
           f"{R['ar0_t']:+.2f} with a leave-one-out range of [{R['jk_ar0_lo']:+.2f}, "
           f"{R['jk_ar0_hi']:+.2f}]; the tradable window *t* = {R['win_t']:+.2f} and the "
           f"six-month *t* = {R['m6_t']:+.2f}, both with bootstrap CIs straddling zero."),
        md("## 3. How do we know the jump is not just noise?\n\n"
           "Three ways.\n\n"
           f"**It is date-locked.** Move the assumed announcement day by one session and the "
           f"effect vanishes: {R['shift_m1']:+.2f}% at day −1, **{R['ar0']:+.2f}% at day 0**, "
           f"{R['shift_p1']:+.2f}% at day +1, {R['shift_p5']:+.2f}% at day +5. Whatever is "
           "happening happens *exactly* when the filing lands.\n\n"
           f"**A placebo cannot fake it.** Re-run the same {R['n_events']} companies on "
           f"randomly chosen dates on their own price histories, 2,000 times: the random "
           f"version averages **{R['pb_ar0_mean']:+.2f}%**. The real one is "
           f"**{R['ar0']:+.2f}%** — *p* = {R['pb_ar0_p']:.4f}.\n\n"
           f"**It survives the split.** {R['era_e_ar0']:+.2f}% in 2010–2017 and "
           f"{R['era_l_ar0']:+.2f}% in 2018–2025. Both halves, same sign, both convincing."),
        md("## 4. So why can't you trade it?\n\n"
           "Because you are not in the room. The offer *becomes* public on the day it moves — "
           "there is no window in which you know about the auction and the price has not "
           "already adjusted. Everything after that is flat.\n\n"
           "We built the obvious sleeve anyway: buy each issuer one session after the filing, "
           "hold to expiry, short the S&P against it. Over "
           f"{R['cal_days']:,} sessions that portfolio compounds to **{R['cal_n_cum']:.1f}%** "
           f"after costs, with a Sharpe of **{R['cal_n_sharpe']:+.3f}**. A long-only version "
           f"that just holds the event names and sits in T-bills otherwise earns an "
           f"excess-of-cash Sharpe of **{R['race_n']:+.3f}** against SPY's "
           f"**{R['race_spy']:+.3f}** — you would have been better off owning the index and "
           "never reading a filing.\n\n"
           "> 🔬 **For the quants.** HAC *t* on the daily *mean* difference between the net "
           f"event basket and SPY, both excess of BIL: **{R['race_t']:+.2f}** — right on the "
           "two-sigma line. The wide number is the Sharpe gap; the mean-difference test only "
           "just calls it at conventional size, and we say so rather than rounding it up."),
        md("## 5. Is the effect at least bigger where you *could* trade?\n\n"
           "No — it is the other way round, as it usually is. Restrict to the "
           f"{R['liq_n']} events on names trading at least $10m a day and the announcement "
           f"pop shrinks from {R['ar0']:.2f}% to **{R['liq_ar0']:.2f}%**. The biggest jumps "
           "belong to the smallest, thinnest issuers — where a tender for 10% of the float is "
           "genuinely transformative and where your own order would move the price."),
        md("## 6. A live check that the machinery works (offline synthetic)\n\n"
           "**This cell does not touch the real tape.** It builds an artificial world where "
           "we *plant* both an announcement pop and a six-month drift, then a second world "
           "with neither, and checks the same code finds the first and stays quiet on the "
           "second. If the harness could not see a planted drift, the flat real-tape drift "
           "would prove nothing."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from dutch_auction import data, strategy as st\n"
            "\n"
            "pl_px, pl_ev, truth = data.synthetic_panel(n_events=60, n_days=900,\n"
            "                                           signal_strength=1.0, seed=927)\n"
            "nl_px, nl_ev, _ = data.synthetic_panel(n_events=60, n_days=900,\n"
            "                                       signal_strength=0.0, seed=927)\n"
            "pl = st.synthetic_detect(pl_px, pl_ev, market='MKT')\n"
            "nl = st.synthetic_detect(nl_px, nl_ev, market='MKT')\n"
            "print('SYNTHETIC (not the real tape)')\n"
            "print('  planted world: jump planted %+.0f bps -> recovered %+.0f bps (t=%+.2f)'\n"
            "      % (truth['planted_jump']*1e4, pl['ar0_bps'], pl['t_ar0']))\n"
            "print('  planted world: 6m drift planted %+.0f bps of log return, which is %+.0f bps'\n"
            "      % (truth['planted_drift_6m_log']*1e4, truth['expected_simple_drift_6m']*1e4))\n"
            "print('                 of expected buy-and-hold return -> recovered %+.0f bps (t=%+.2f)'\n"
            "      % (pl['m6_bps'], pl['t_m6']))\n"
            "print('  null world   : day-0 %+.0f bps (t=%+.2f)  <- must be ~0'\n"
            "      % (nl['ar0_bps'], nl['t_ar0']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** Two claims, two answers. The announcement repricing is "
           f"**real and large** ({R['ar0']:+.2f}%, *t* = {R['ar0_t']:+.2f}, "
           f"{R['ar0_hit']}% of {R['n_events']} events, placebo *p* = {R['pb_ar0_p']:.4f}, "
           f"date-locked, present in both eras). The \"it marks the bottom\" claim is "
           f"**absent**: the tender window pays {R['win']:+.2f}% (*t* = {R['win_t']:+.2f}) and "
           f"six months later the median company is {R['m6_med']:+.2f}% behind the market.\n"
           f"- **Tradability — Mirage.** The one day that pays is the one day you cannot be "
           f"positioned for. Everything after it is zero gross and negative net: "
           f"{R['cost10_win']:+.2f}% per event at 10 bps, a {R['cal_n_sharpe']:+.3f} Sharpe "
           f"for the long/short sleeve, and {R['race_adv']:+.3f} Sharpe *behind* simply "
           f"owning SPY."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 927 — Dutch Auction — the teardown\n\n"
           f"{R['n_events']} issuer *modified Dutch auction* self-tenders on "
           f"{R['n_issuers']} distinct issuers, {R['start']} → {R['end']}. Event set built by "
           "one EDGAR full-text-search query (`q=\"modified Dutch auction\"`, "
           "`forms=SC TO-I`), clustered per registrant, event date = the earliest SC TO-I "
           "filing of each cluster (the commencement date, public that session). Abnormal "
           "return = issuer − SPY on daily **total-return** closes (`auto_adjust=True`). One "
           "execution lag: filing public at the close of *t*, tradable leg entered at the "
           "close of *t+1*.\n\n"
           f"Every real number is frozen from `docs/results.md` (Fingerprint `{R['fp']}`); the "
           "live cells at the end are the **synthetic** control."),
        code("R = %r" % (R,)),
        md("## The window decomposition\n\n"
           "Four separate objects, deliberately never blended: the pre-announcement run-up "
           "(SC TO-C / press-release leakage), the announcement-day repricing (news, not an "
           "edge), the tradable tender window, and the post-expiry drift.\n\n"
           "> 💡 **In plain words.** We split the event into \"what happened before anyone "
           "could act\", \"what happened the instant it became public\", and \"what happened "
           "afterwards, when you could actually buy\"."),
        code(
            "rows = [('pre5  (-5,-1) run-up', R['pre5'], R['pre5_med'], R['pre5_t'], R['pre5_hit']),\n"
            "        ('ar0   day 0 (news)  ', R['ar0'],  R['ar0_med'],  R['ar0_t'],  R['ar0_hit']),\n"
            "        ('window t+1 -> expiry', R['win'],  R['win_med'],  R['win_t'],  R['win_hit']),\n"
            "        ('m1    expiry +1m    ', R['m1'],   R['m1_med'],   R['m1_t'],   R['m1_hit']),\n"
            "        ('m3    expiry +3m    ', R['m3'],   R['m3_med'],   R['m3_t'],   R['m3_hit']),\n"
            "        ('m6    expiry +6m    ', R['m6'],   R['m6_med'],   R['m6_t'],   R['m6_hit'])]\n"
            "print(f\"n = {R['n_events']} events on {R['n_issuers']} issuers\\n\")\n"
            "print(f\"{'window':22s} {'mean':>8s} {'median':>8s} {'t':>7s} {'hit':>5s}\")\n"
            "for lab, mu, med, t, hit in rows:\n"
            "    print(f'{lab:22s} {mu:+7.2f}% {med:+7.2f}% {t:+7.2f} {hit:4d}%')"
        ),
        md("## Is day 0 robust? Jackknife, block bootstrap, placebo, date shift\n\n"
           "Four checks, each attacking a different failure mode: one lucky event, a fat "
           "tail, a mis-specified null, and a mis-dated event."),
        code(
            "print('jackknife (leave-one-out t)')\n"
            "print(f\"  ar0    t={R['ar0_t']:+.2f}  LOO [{R['jk_ar0_lo']:+.2f}, {R['jk_ar0_hi']:+.2f}]\")\n"
            "print(f\"  window t={R['win_t']:+.2f}  LOO [{R['jk_win_lo']:+.2f}, {R['jk_win_hi']:+.2f}]\")\n"
            "print(f\"  m6     t={R['m6_t']:+.2f}  LOO [{R['jk_m6_lo']:+.2f}, {R['jk_m6_hi']:+.2f}]\")\n"
            "print('\\nblock bootstrap on the mean (5,000 draws, 5-event blocks)')\n"
            "print(f\"  ar0    {R['ar0']:+.2f}%  CI [{R['ci_ar0_lo']:+.2f}%, {R['ci_ar0_hi']:+.2f}%]\")\n"
            "print(f\"  window {R['win']:+.2f}%  CI [{R['ci_win_lo']:+.2f}%, {R['ci_win_hi']:+.2f}%]  \"\n"
            "      f\"({R['ci_win_neg']:.1f}% of draws negative)\")\n"
            "print(f\"  m6     {R['m6']:+.2f}%  CI [{R['ci_m6_lo']:+.2f}%, {R['ci_m6_hi']:+.2f}%]  \"\n"
            "      f\"({R['ci_m6_neg']:.1f}% negative)\")\n"
            "print('\\nplacebo: same names, dates re-drawn at random on their own tapes (2,000 draws)')\n"
            "print(f\"  ar0    obs {R['ar0']:+.2f}%  placebo {R['pb_ar0_mean']:+.2f}% \"\n"
            "      f\"(sd {R['pb_ar0_sd']:.2f}%)  p={R['pb_ar0_p']:.4f}\")\n"
            "print(f\"  window p={R['pb_win_p']:.4f}   m6 p={R['pb_m6_p']:.4f}   <- both indistinguishable from chance\")\n"
            "print('\\nevent-date shift (trading days)')\n"
            "print(f\"  -5: {R['shift_m5']:+.2f}%   -1: {R['shift_m1']:+.2f}% (t={R['shift_m1_t']:+.2f})   \"\n"
            "      f\"0: {R['ar0']:+.2f}% (t={R['ar0_t']:+.2f})   +1: {R['shift_p1']:+.2f}% \"\n"
            "      f\"(t={R['shift_p1_t']:+.2f})   +5: {R['shift_p5']:+.2f}%\")"
        ),
        md("## The overlapping-event problem, and the calendar-time fix\n\n"
           "Events cluster in calendar time (2015 and 2021 were busy), so the cross-event "
           "one-sample *t* is not correctly sized at long horizons. The standard fix "
           "(Mitchell & Stafford 2000) is a calendar-time portfolio: equal-weight every name "
           "currently inside its window, take the daily series, and use a HAC *t*.\n\n"
           "> 💡 **In plain words.** If ten of your events happen in the same quarter, they "
           "are really one bet on that quarter, not ten independent bets. The calendar-time "
           "portfolio counts it as one.\n\n"
           "**How the costs are charged, and why it matters.** The sleeve is equal-weight "
           "across whichever names are live, so a slot sharing the book with two others is a "
           "third of NAV and can only cost a third of a round trip. Costs are therefore "
           "booked at each slot's **real portfolio weight**, on the day it opens and the day "
           f"it closes: the {R['n_events']} events come to **{R['cal_rt']:.1f}** full-NAV "
           f"round trips, a drag of {R['cal_drag']:.2f} bps/day. Charging one full-NAV round "
           f"trip per event — the obvious shortcut — would have inflated that by "
           f"**{R['cal_overstate']:.2f}×** and handed the negative verdict to the cost model "
           "instead of the tape. It does not come to that: the sleeve is negative "
           f"**{R['cal_g_sharpe']:+.3f} gross**, before a single basis point is charged."),
        code(
            "print('long issuer / short SPY, entered t+1, held to the expiry proxy')\n"
            "print(f\"  gross: {R['cal_g_bps']:+.3f} bps/day  Sharpe {R['cal_g_sharpe']:+.3f}  \"\n"
            "      f\"HAC t {R['cal_g_t']:+.2f}  cum {R['cal_g_cum']:+.2f}%\")\n"
            "print(f\"  net  : {R['cal_n_bps']:+.3f} bps/day  Sharpe {R['cal_n_sharpe']:+.3f}  \"\n"
            "      f\"HAC t {R['cal_n_t']:+.2f}  cum {R['cal_n_cum']:+.2f}%   (10 bps/leg, 30 bps/yr borrow)\")\n"
            "print(f\"  live on {R['cal_live']:,} of {R['cal_days']:,} sessions, \"\n"
            "      f\"{R['cal_names']:.2f} names on an average live day, vol {R['cal_vol']:.1f}%\")\n"
            "print('\\nlong-only race, excess-of-cash (BIL) vs SPY')\n"
            "print(f\"  event basket gross : exSharpe {R['race_g']:+.3f}\")\n"
            "print(f\"  event basket net   : exSharpe {R['race_n']:+.3f}  vol {R['race_n_vol']:.1f}%\")\n"
            "print(f\"  SPY                : exSharpe {R['race_spy']:+.3f}  vol {R['race_spy_vol']:.1f}%  \"\n"
            "      f\"HAC t {R['race_spy_t']:+.2f}\")\n"
            "print(f\"  advantage {R['race_adv']:+.3f}   HAC t on the daily difference {R['race_t']:+.2f}\")"
        ),
        md("## Era cut, liquidity cut, and the sweeps\n\n"
           "The announcement effect must survive an era split and the liquid subset; the "
           "tradable legs must survive nothing, because there is nothing there. Every "
           "assumption that is not a tape input — cost, borrow, the expiry proxy — is swept."),
        code(
            "print('era cut (split 2018-01-01)')\n"
            "print(f\"  2010-2017 (n={R['era_e_n']}): ar0 {R['era_e_ar0']:+.2f}% (t={R['era_e_ar0_t']:+.2f})  \"\n"
            "      f\"window {R['era_e_win']:+.2f}% (t={R['era_e_win_t']:+.2f})  m6 {R['era_e_m6']:+.2f}% (t={R['era_e_m6_t']:+.2f})\")\n"
            "print(f\"  2018-2025 (n={R['era_l_n']}): ar0 {R['era_l_ar0']:+.2f}% (t={R['era_l_ar0_t']:+.2f})  \"\n"
            "      f\"window {R['era_l_win']:+.2f}% (t={R['era_l_win_t']:+.2f})  m6 {R['era_l_m6']:+.2f}% (t={R['era_l_m6_t']:+.2f})\")\n"
            "print(f\"\\nliquidity subset (median 60d dollar volume >= $10m, n={R['liq_n']})\")\n"
            "print(f\"  ar0 {R['liq_ar0']:+.2f}% (t={R['liq_ar0_t']:+.2f}, hit {R['liq_ar0_hit']}%)  \"\n"
            "      f\"window {R['liq_win']:+.2f}% (t={R['liq_win_t']:+.2f})  \"\n"
            "      f\"m6 {R['liq_m6']:+.2f}% (median {R['liq_m6_med']:+.2f}%, t={R['liq_m6_t']:+.2f})\")\n"
            "print(f\"  calendar-time net Sharpe {R['liq_cal_sharpe']:+.3f} (HAC t {R['liq_cal_t']:+.2f})\")\n"
            "print('\\ncost sweep (one-way x NAV, both legs, in and out)')\n"
            "for c, w, s in [(0, R['cost0_win'], R['cost0_sharpe']), (10, R['cost10_win'], R['cost10_sharpe']),\n"
            "                (25, R['cost25_win'], R['cost25_sharpe']), (50, R['cost50_win'], R['cost50_sharpe'])]:\n"
            "    print(f'  {c:2d} bps: net window {w:+.2f}%   calendar-time Sharpe {s:+.3f}')\n"
            "print(f\"\\nborrow sweep (short-SPY leg, an ASSUMPTION): Sharpe {R['borrow0']:+.3f} at 0 bps/yr \"\n"
            "      f\"-> {R['borrow300']:+.3f} at 300 bps/yr\")\n"
            "print('\\nexpiry PROXY sweep (Rule 14e-1 minimum = 20 sessions)')\n"
            "print(f\"  +15td: window {R['exp15_win']:+.2f}%  m6 {R['exp15_m6']:+.2f}%\")\n"
            "print(f\"  +20td: window {R['win']:+.2f}%  m6 {R['m6']:+.2f}%   <- headline\")\n"
            "print(f\"  +25td: window {R['exp25_win']:+.2f}%  m6 {R['exp25_m6']:+.2f}%\")\n"
            "print(f\"  +30td: window {R['exp30_win']:+.2f}%  m6 {R['exp30_m6']:+.2f}%\")"
        ),
        md("## Live synthetic control — recover the plant, stay quiet on the null\n\n"
           "**Synthetic, not the real tape.** 60 issuers on one shared market factor. The "
           "planted world carries a day-0 jump and a six-month drift; the null carries "
           "neither. The harness must recover the first and find nothing in the second."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from dutch_auction import data, strategy as st\n"
            "\n"
            "px, ev, truth = data.synthetic_panel(n_events=60, n_days=900,\n"
            "                                     signal_strength=1.0, seed=927)\n"
            "d = st.synthetic_detect(px, ev, market='MKT')\n"
            "print('SYNTHETIC planted world (not the real tape)')\n"
            "print('  day-0 : planted %+.0f bps -> recovered %+.0f bps (t=%+.2f)'\n"
            "      % (truth['planted_jump']*1e4, d['ar0_bps'], d['t_ar0']))\n"
            "print('  6m    : planted %+.0f bps of LOG drift = %+.0f bps of expected SIMPLE'\n"
            "      % (truth['planted_drift_6m_log']*1e4, truth['expected_simple_drift_6m']*1e4))\n"
            "print('          buy-and-hold abnormal return (Jensen) -> recovered %+.0f bps (t=%+.2f)'\n"
            "      % (d['m6_bps'], d['t_m6']))\n"
            "\n"
            "ts = []\n"
            "for s in range(12):\n"
            "    npx, nev, _ = data.synthetic_panel(n_events=60, n_days=900,\n"
            "                                       signal_strength=0.0, seed=927+s)\n"
            "    ts.append(st.synthetic_detect(npx, nev, market='MKT')['t_ar0'])\n"
            "ts = np.array(ts)\n"
            "print('SYNTHETIC null x12: mean t(day-0) %+.2f, max |t| %.2f, |t|>=2 on %d/12'\n"
            "      % (ts.mean(), np.abs(ts).max(), (np.abs(ts) >= 2).sum()))"
        ),
        md(f"## A finding hiding in the null\n\n"
           f"On the headline run the null was pushed to {R['syn_null_n']} seeds, and the "
           f"day-0 statistic behaved (|*t*| max {R['syn_null_ar0_max']:.2f}, ≥2 on "
           f"{R['syn_null_ar0_fire']}/{R['syn_null_n']}) while the **six-month** statistic "
           f"did not: |*t*| max {R['syn_null_m6_max']:.2f}, ≥2 on "
           f"{R['syn_null_m6_fire']}/{R['syn_null_n']}. That is not a bug — it is Barber-Lyon "
           "and Kothari-Warner reproduced in miniature. A six-month *simple* buy-and-hold "
           "abnormal return is right-skewed and positively centred even under a null (the "
           "same Jensen term that turns a planted "
           f"{R['syn_drift']} bps of log drift into {R['syn_drift_exp']} bps of expected "
           "simple return), and the shared market factor makes those returns "
           "cross-sectionally correlated on top. So the naive cross-event *t* "
           "**over-rejects** at long horizons. The correctly sized tests for the "
           f"drift are therefore the placebo (*p* = {R['pb_m6_p']:.4f}) and the calendar-time "
           f"HAC *t* ({R['cal_n_t']:+.2f}) — both empty — and the real-tape six-month *t* of "
           f"{R['m6_t']:+.2f} is, if anything, flattered."),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The **announcement repricing is Real**: "
           f"{R['ar0']:+.2f}% abnormal on the SC TO-I session, one-sample *t* = "
           f"{R['ar0_t']:+.2f}, jackknife LOO [{R['jk_ar0_lo']:+.2f}, {R['jk_ar0_hi']:+.2f}], "
           f"bootstrap CI [{R['ci_ar0_lo']:+.2f}%, {R['ci_ar0_hi']:+.2f}%], placebo "
           f"*p* = {R['pb_ar0_p']:.4f}, date-locked to a single session, present in both eras "
           f"({R['era_e_ar0']:+.2f}% / {R['era_l_ar0']:+.2f}%) and in the liquid subset "
           f"({R['liq_ar0']:+.2f}%, *t* = {R['liq_ar0_t']:+.2f}). The **bottom-marking claim "
           f"is None**: tender window {R['win']:+.2f}% (*t* = {R['win_t']:+.2f}, placebo "
           f"*p* = {R['pb_win_p']:.2f}), six-month drift {R['m6']:+.2f}% with a "
           f"{R['m6_med']:+.2f}% median (*t* = {R['m6_t']:+.2f}, CI "
           f"[{R['ci_m6_lo']:+.2f}%, {R['ci_m6_hi']:+.2f}%]). Named biases: the CIK→ticker map "
           f"is SEC's **current** register (survivorship); the screens need 148 sessions of "
           f"tape AFTER the event, so an issuer taken out within ~7 months of its own tender "
           f"is dropped — exactly the outcome that would print a big positive six-month "
           f"drift, which makes the flat drift above an **upper bound**; and the sample is "
           f"the subset of self-tenders whose filings say \"modified Dutch auction\" "
           f"(visibility).\n"
           f"- **Tradability — Mirage.** The paying session is the one you cannot be "
           f"positioned for. The calendar-time long/short compounds to {R['cal_n_cum']:.1f}% "
           f"net ({R['cal_n_sharpe']:+.3f} Sharpe, HAC *t* {R['cal_n_t']:+.2f}); the long-only "
           f"basket trails SPY by {R['race_adv']:+.3f} Sharpe with a HAC *t* of "
           f"{R['race_t']:+.2f} on the daily difference; cost and borrow sweeps only make it "
           f"worse; no expiry assumption rescues it. The premium is paid to the holders who "
           f"tender, not to the tape."),
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
