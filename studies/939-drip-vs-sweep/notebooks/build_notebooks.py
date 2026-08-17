"""Generate the two narrative notebooks for Study 939 (DRIP or Sweep).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the
fast synthetic control and the offline reconstruction check on a synthetic tape, so
execution is quick and network-free. No synthetic cell ever appears under a real-tape
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


# --------------------------------------------------------------------------- #
# Frozen real-tape headline — mirror of docs/results.md.
# SPY/VYM/SCHD price-only + reconstructed distributions vs BIL, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    asof="2026-06-30", fp="138ed0f07aec",
    start="2007-05-30", end="2026-06-30", n_spy=4801, n_schd=3692,
    pay_lag=30, drip_cost=0.0, sweep_cost=2.0,

    # reconstruction audit, on the RACED window (the shared cache's pre-window history
    # is written by whichever study fetched a ticker first and is not ours to pin)
    rec=dict(SPY=(77, 77, 77, 88.667, 88.854, 0.9979),
             VYM=(77, 77, 77, 45.200, 45.186, 1.0003),
             SCHD=(59, 59, 59, 8.970, 8.985, 0.9984)),
    tr_audit=dict(SPY=(0.99972, 0.042, -0.15), VYM=(0.99932, 0.068, -0.36),
                  SCHD=(0.99889, 0.111, -0.75)),

    # headline race, quarterly sweep. `yld` is the REALISED in-sample distribution
    # yield over each fund's own window, not the fund's current headline yield.
    spy=dict(years=19.1, sh_drip=0.5420, sh_sweep=0.5424, w_drip=69003, w_sweep=68729,
             gap=2.09, t=1.09, ci=(-1.18, 4.99), yld=1.86),
    vym=dict(years=19.1, sh_drip=0.4865, sh_sweep=0.4867, w_drip=51370, w_sweep=51033,
             gap=3.46, t=1.18, ci=(-1.75, 7.97), yld=3.12),
    schd=dict(years=14.7, sh_drip=0.7785, sh_sweep=0.7801, w_drip=60195, w_sweep=59863,
              gap=3.78, t=1.41, ci=(-1.74, 8.30), yld=3.20),

    # the same race on SCHD's window for all three (yield, gap, t, gap per 1% of yield)
    matched=dict(SPY=(1.77, 3.01, 1.87, 1.70), VYM=(3.09, 3.73, 1.40, 1.21),
                 SCHD=(3.20, 3.78, 1.41, 1.18)),
    # HAC lag robustness of the headline t: (auto, 63-day, 252-day)
    hac=dict(SPY=(1.09, 1.33, 1.57), VYM=(1.18, 1.42, 1.60), SCHD=(1.41, 1.46, 1.57)),
    pooled_q=dict(gap=2.81, t=1.15, ci=(-1.47, 6.51), neg=0.084),
    pooled_a=dict(gap=12.78, t=1.94, ci=(0.04, 23.41), neg=0.025),

    # annual sweep
    annual=dict(SPY=(9.18, 1.71), VYM=(15.46, 2.02), SCHD=(17.77, 2.73)),

    # era cut
    era=dict(SPY=((1.68, 0.48), (1.84, 0.91)), VYM=((3.55, 0.71), (2.43, 0.71)),
             SCHD=((3.42, 1.09), (3.18, 0.91))),
    # rate-regime cut (low, high)
    rate=dict(SPY=((3.65, 1.28, 15), (6.05, 2.32, 4)),
              VYM=((5.09, 1.28, 15), (8.49, 1.91, 4)),
              SCHD=((5.61, 1.56, 11), (8.53, 1.34, 4))),
    # pay-lag sweep (0, 15, 30, 45)
    lag=dict(SPY=(0.72, 3.63, 2.09, 1.23), VYM=(0.89, 4.11, 3.46, 1.59),
             SCHD=(0.80, 6.96, 3.78, 2.47)),
    # cost sweep (0,0) (0,2) (0,5) (0,10) (5,5)
    cost=dict(SPY=(2.05, 2.09, 2.14, 2.23, 2.05), VYM=(3.40, 3.46, 3.54, 3.69, 3.39),
              SCHD=(3.72, 3.78, 3.87, 4.01, 3.71)),

    # money
    money=dict(SPY=(274, 14.4), VYM=(337, 17.7), SCHD=(332, 22.7)),

    # synthetic control (lab bench, 8 seeds)
    syn_q_planted=(14.23, 1.60, 0.56), syn_q_null=(0.03, 1.71, 0.60),
    syn_a_planted=(52.13, 5.28, 1.87), syn_a_null=(-1.03, 6.12, 2.16),
    syn_mono=((3, 8.80), (6, 13.72), (9, 20.88)),
    power_planted=(1.50, 2.26, 0.80), power_null=(-0.52, 2.30, 0.81),
)


HEADER = f"""# Study 939 — DRIP or Sweep 💧

**Reinvest each dividend the day it lands, or sweep it to cash and reinvest on a schedule?**

Every broker's help page tells you to tick the automatic-reinvestment box, because idle
cash is cash not compounding. That is a testable claim about a *timing difference on a
small cash flow*, and nobody who repeats it seems to have put a number on it.

We put one on it. Two investors, the **same fund**, the **same shares on day one**.
**DRIP** buys more shares on the pay date. **SWEEP** parks the cash in T-bills (BIL) and
buys at the next quarter end. Everything else — one execution lag, one-way costs on the
amount traded, excess-of-cash Sharpes — is identical.

Tapes: **SPY, VYM, SCHD** against **BIL**, {R['start']} → {R['end']}
({R['n_spy']:,} days; SCHD from 2011). The distribution stream is **reconstructed** from
the price-only and total-return legs.

*Real numbers below are the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`, as-of {R['asof']}); the live cells run the offline synthetic control only.*
"""


def build_curious():
    nb = new_notebook()
    s, v, c = R["spy"], R["vym"], R["schd"]
    cells = [
        md(HEADER),
        md("## 1. Where the money can even come from\n\n"
           "A dividend leaves the share price on the **ex-date** and lands in your account "
           "on the **pay date**, two to five weeks later. From that moment it is either "
           "buying shares (DRIP) or sitting in T-bills waiting for the calendar (SWEEP).\n\n"
           "So the entire prize is: *what the fund earns over cash, on that money, for the "
           "few weeks the sweeper is waiting*. Multiply it out — a 2% yield, a 5-point "
           "equity-over-cash spread, a six-week wait — and you get about **one basis point "
           "a year**. That is the size of the thing before we touch any data.\n\n"
           "> 🔬 **For the quants:** the gap is (distribution yield) × (realised "
           "equity-minus-cash return over the delay) × (average delay). Everything else in "
           "this study is measurement error around that product."),
        md("## 2. What the tape actually pays\n\n"
           "Nineteen years of SPY, VYM and SCHD, with the dividend stream rebuilt from the "
           "price and total-return series and every reinvestment costed:"),
        code(
            "R = " + repr(dict(spy=R["spy"], vym=R["vym"], schd=R["schd"],
                               matched=R["matched"])) + "\n"
            "for tag, d in [('SPY ', R['spy']), ('VYM ', R['vym']), ('SCHD', R['schd'])]:\n"
            "    print(f\"{tag} (realised yield {d['yld']:.2f}%): DRIP ends at {d['w_drip']:,} vs \"\n"
            "          f\"SWEEP {d['w_sweep']:,} on 10,000 invested over {d['years']} years\")\n"
            "    print(f\"       -> DRIP wins by {d['gap']:+.2f} basis points a year \"\n"
            "          f\"(HAC t = {d['t']:+.2f})\")"
        ),
        md(f"## 3. Three basis points, in money\n\n"
           f"On **10,000** invested, ticking the DRIP box instead of sweeping quarterly was "
           f"worth **{R['money']['SPY'][0]:,} over {s['years']} years on SPY** — about "
           f"**{R['money']['SPY'][1]:.0f} a year**. On the dividend funds it is a bit more: "
           f"**{R['money']['VYM'][1]:.0f}/yr** (VYM) and **{R['money']['SCHD'][1]:.0f}/yr** "
           f"(SCHD). That is less than the half-spread on a single ETF trade.\n\n"
           f"It is tempting to read the ordering — {s['gap']:.2f} bps on the "
           f"{s['yld']:.2f}%-yield SPY, {v['gap']:.2f} on VYM, {c['gap']:.2f} on SCHD — as "
           f"the mechanism showing its face: more cash in transit, more to lose by parking "
           f"it. Careful. SPY and VYM are measured over 2007-2026 and SCHD only over "
           f"2011-2026, and when all three are re-raced on the *same* window the spread "
           f"nearly disappears ({R['matched']['SPY'][1]:.2f} / {R['matched']['VYM'][1]:.2f} "
           f"/ {R['matched']['SCHD'][1]:.2f} bps/yr). The mechanism is real arithmetic; the "
           f"cross-fund pattern is mostly calendar."),
        md(f"## 4. The one version that costs real money\n\n"
           f"Sweeping **quarterly** costs almost nothing, because the pay-date lag has "
           f"already eaten most of the delay by the time the quarter ends. Sweeping "
           f"**annually** — letting a year of distributions pile up in cash before "
           f"reinvesting — is a different story:\n\n"
           f"| Fund | quarterly sweep | annual sweep |\n|---|--:|--:|\n"
           f"| SPY | {s['gap']:+.2f} bps/yr | **{R['annual']['SPY'][0]:+.2f}** bps/yr |\n"
           f"| VYM | {v['gap']:+.2f} bps/yr | **{R['annual']['VYM'][0]:+.2f}** bps/yr |\n"
           f"| SCHD | {c['gap']:+.2f} bps/yr | **{R['annual']['SCHD'][0]:+.2f}** bps/yr |\n\n"
           f"Still small — but 15 to 18 basis points a year is at least the size of a fund "
           f"fee, and it is the only cut in this study where the numbers clear the desk's "
           f"significance bar (SCHD *t* = {R['annual']['SCHD'][1]:+.2f})."),
        md(f"## 5. The number we cannot see\n\n"
           f"Here is the honest problem. Yahoo publishes the **ex-date**; nobody free "
           f"publishes the **pay date**. We assumed {R['pay_lag']} calendar days. Change "
           f"that assumption and the answer moves like this (SCHD):\n\n"
           f"| assumed pay lag | 0 days | 15 days | 30 days | 45 days |\n|---|--:|--:|--:|--:|\n"
           f"| gap | {R['lag']['SCHD'][0]:+.2f} | {R['lag']['SCHD'][1]:+.2f} | "
           f"{R['lag']['SCHD'][2]:+.2f} | {R['lag']['SCHD'][3]:+.2f} bps/yr |\n\n"
           f"**The thing we had to guess moves the answer more than the answer is worth.** "
           f"That is the sentence to remember from this study."),
        md("## 6. Live check — is the measuring stick straight? (offline synthetic)\n\n"
           "Before believing a two-basis-point result, check the machinery can find a "
           "*planted* one and stays quiet when there is nothing there. On a deliberately "
           "loud laboratory tape (a big premium over cash, a fat dividend, low noise) the "
           "DRIP arm must win; on a tape where the fund drifts at exactly the cash rate, "
           "parking costs nothing and the gap must vanish."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from drip_sweep import data, strategy as st\n"
            "planted = st.seed_sweep(data.synthetic_daily, 1.0, n_seeds=8)\n"
            "null    = st.seed_sweep(data.synthetic_daily, 0.0, n_seeds=8)\n"
            "print('lab tape, planted premium : gap %+.2f bps/yr (se %.2f)  <- must be clearly positive'\n"
            "      % (planted['mean'], planted['se']))\n"
            "print('lab tape, no premium      : gap %+.2f bps/yr (se %.2f)  <- must be ~0'\n"
            "      % (null['mean'], null['se']))"
        ),
        md("## 7. And the check that decides the verdict\n\n"
           "Now feed the *same* measuring stick a **market-realistic** world — a 5.5% "
           "premium, a 3% yield, 16% volatility — and ask it to find the true effect in "
           "twenty years of daily data. It cannot. The noise from one twenty-year path is "
           "bigger than the thing being measured."),
        code(
            "real = dict(equity_premium=0.055, div_yield_ann=0.03, vol_ann=0.16)\n"
            "pl = st.seed_sweep(data.synthetic_daily, 1.0, n_seeds=8, gen_kw=real)\n"
            "nl = st.seed_sweep(data.synthetic_daily, 0.0, n_seeds=8, gen_kw=real)\n"
            "print('realistic world, true effect present: %+.2f bps/yr (se %.2f)' % (pl['mean'], pl['se']))\n"
            "print('realistic world, no effect         : %+.2f bps/yr (se %.2f)' % (nl['mean'], nl['se']))\n"
            "print('\\nOne 20-year tape cannot tell these two apart. That is why the real-tape')\n"
            "print('t-statistic is ~1.2 and not ~5: the effect is real and it is tiny.')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** DRIP wins on all three funds and in both eras "
           f"({s['gap']:+.2f} / {v['gap']:+.2f} / {c['gap']:+.2f} bps/yr) — the right sign "
           f"everywhere. But the *t*-statistics are {s['t']:+.2f} / {v['t']:+.2f} / "
           f"{c['t']:+.2f}, every confidence interval includes zero, the yield ordering "
           f"mostly evaporates once the three funds are raced on the same window, and the "
           f"unobservable pay-date lag swings the estimate across a wider band than the "
           f"estimate itself. Only the annual-sweep variant clears the desk's |*t*| ≥ 2 "
           f"bar.\n"
           f"- **Tradability — Mirage.** {R['pooled_q']['gap']:+.2f} basis points a year is "
           f"{R['money']['SPY'][1]:.0f}-{R['money']['SCHD'][1]:.0f} a year on 10,000 — less "
           f"than one trade's spread. Tick the DRIP box because it is free and saves you "
           f"from making a decision four times a year. If you prefer to sweep, do it "
           f"**quarterly, not annually**: that is the only version of this choice that "
           f"costs measurable money."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    s, v, c = R["spy"], R["vym"], R["schd"]
    cells = [
        md("# Study 939 — DRIP or Sweep — the teardown\n\n"
           "The dividend reconstruction and its two audits, the terminal-wealth gap with a "
           "Newey-West *t* on the daily log-return difference, 63-day circular block "
           "bootstrap CIs, an era cut, a rate-regime cut, sweeps of every labelled "
           "assumption, and a power calculation on the synthetic control.\n\n"
           f"Every real number is frozen from `docs/results.md` (Fingerprint `{R['fp']}`, "
           f"as-of {R['asof']}). Live cells are synthetic only and are labelled as such.\n\n"
           "**Design.** Two arms over one holding. Shares are bought once at *t*₀ with "
           "10,000; thereafter the arms differ only in what happens to a distribution. "
           "Exactly one execution lag: a purchase decided at the close of *t* is filled at "
           "the close of *t*+1. Costs are one-way × the amount reinvested (never × NAV); "
           "neither arm ever shorts, so no borrow applies. Sharpes are excess-of-cash "
           "(minus BIL's total return) on **both** arms. The simulation runs on the "
           "**price-only** close; the **total-return** close is used only to build the "
           "distribution stream and to audit it.\n\n"
           "> 💡 **In plain words:** same fund, same shares, one difference — do you buy "
           "the moment the dividend lands, or let it sit in T-bills until quarter end?"),
        code("R = " + repr(R)),
        md("## 1. The reconstruction\n\n"
           "`D_t = P_{t−1}·(TR_t / TR_{t−1}) − P_t`, thresholded at 5 bp of price to reject "
           "the float dust that survives the two legs' independent rounding. The vendor's "
           "reported cash column is used **only** to score this — never as an input.\n\n"
           "> 💡 **In plain words:** the adjusted-close series already contains the "
           "dividends. Divide it by the unadjusted one and the payments fall out."),
        code(
            "for tk in ('SPY','VYM','SCHD'):\n"
            "    n_rec, n_rep, n_match, tot_rec, tot_rep, ratio = R['rec'][tk]\n"
            "    tr_ratio, dev, track = R['tr_audit'][tk]\n"
            "    print(f\"{tk:5s} events {n_rec:3d} rec / {n_rep:3d} reported / {n_match:3d} matched  |  \"\n"
            "          f\"cash/share {tot_rec:8.3f} vs {tot_rep:8.3f} (ratio {ratio:.4f})\")\n"
            "    print(f\"      zero-lag DRIP vs the TR index: terminal ratio {tr_ratio:.5f}, \"\n"
            "          f\"max dev {dev:.3f}%, tracking {track:+.2f} bps/yr\")"
        ),
        md("The second line is the study's load-bearing audit: a **zero-lag, zero-cost DRIP "
           "is the definition of the adjusted close**, so it has to reproduce it. It does, "
           "to within 0.04-0.11% over 15-19 years; the residual is the single execution lag."),
        md("## 2. The headline race — DRIP vs quarterly sweep\n\n"
           "Excess-of-cash Sharpes on both arms, terminal wealth from 10,000, 63-day block "
           "bootstrap (2,000 draws) on the annualised log-wealth gap."),
        code(
            "for tag, d in [('SPY', R['spy']), ('VYM', R['vym']), ('SCHD', R['schd'])]:\n"
            "    lo, hi = d['ci']\n"
            "    print(f\"{tag:5s} {d['years']:.1f}yr  exSharpe DRIP {d['sh_drip']:+.4f} / \"\n"
            "          f\"SWEEP {d['sh_sweep']:+.4f}   W {d['w_drip']:,} vs {d['w_sweep']:,}\")\n"
            "    print(f\"      gap {d['gap']:+.2f} bps/yr  HAC t {d['t']:+.2f}  \"\n"
            "          f\"95% CI [{lo:+.2f}, {hi:+.2f}]\")\n"
            "p = R['pooled_q']\n"
            "print(f\"\\npooled (equal-weight): {p['gap']:+.2f} bps/yr  HAC t {p['t']:+.2f}  \"\n"
            "      f\"CI [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]  share of resamples < 0: {p['neg']:.1%}\")\n"
            "print('note: the excess-of-cash SHARPE race is a dead heat and runs the other way')\n"
            "print('      in the 4th decimal - the swept arm carries a sliver of cash ballast.')"
        ),
        md("Two things the headline table does **not** get to claim.\n\n"
           "First, the *t*-statistics are not an artefact of a short HAC kernel — the "
           "automatic lag (≈9 days) gives the *lowest* reading of the three tested, and even "
           "a 252-day kernel leaves every fund under 1.6.\n\n"
           "Second, the ordering by distribution yield is far weaker than it looks. SPY and "
           "VYM are raced over 2007-2026 and SCHD only over 2011-2026, so the triple mixes "
           "*yield* with *window*. Re-race all three on SCHD's window and the spread "
           "collapses; normalised by realised yield, the ordering **reverses**."),
        code(
            "print('HAC lag robustness of the headline t (auto is the LOWEST reading):')\n"
            "print(f\"{'':7s}{'auto':>8s}{'63d':>8s}{'252d':>8s}\")\n"
            "for tk in ('SPY','VYM','SCHD'):\n"
            "    print('  %-5s' % tk + ''.join('%8.2f' % x for x in R['hac'][tk]))\n"
            "print('\\nsame race on the MATCHED window (SCHD start -> 2026-06-30):')\n"
            "print(f\"{'':7s}{'yield':>8s}{'gap':>8s}{'t':>8s}{'gap/1% yld':>12s}\")\n"
            "for tk in ('SPY','VYM','SCHD'):\n"
            "    y, g, t, per = R['matched'][tk]\n"
            "    print(f\"  {tk:<5s}{y:7.2f}%{g:8.2f}{t:8.2f}{per:12.2f}\")\n"
            "print('-> the ordering survives, the spread does not, and per unit of yield')\n"
            "print('   the LOW-yield fund shows the LARGEST gap. Most of the headline')\n"
            "print('   spread is SPY carrying an extra ZIRP half, not yield.')"
        ),
        md("## 3. Sweep frequency — the only cut with a pulse"),
        code(
            "for tk in ('SPY','VYM','SCHD'):\n"
            "    q = R['spy' if tk=='SPY' else ('vym' if tk=='VYM' else 'schd')]\n"
            "    ga, ta = R['annual'][tk]\n"
            "    print(f\"{tk:5s} quarterly {q['gap']:+6.2f} bps/yr (t {q['t']:+.2f})   \"\n"
            "          f\"annual {ga:+6.2f} bps/yr (t {ta:+.2f})\")\n"
            "pa = R['pooled_a']\n"
            "print(f\"pooled annual: {pa['gap']:+.2f} bps/yr (t {pa['t']:+.2f}), \"\n"
            "      f\"CI [{pa['ci'][0]:+.2f}, {pa['ci'][1]:+.2f}] - lower edge sits ON zero\")"
        ),
        md("> 💡 **In plain words:** waiting until quarter end costs nothing, because the "
           "pay-date lag has already used up most of the delay. Waiting a whole year costs "
           "about as much as a fund fee."),
        md("## 4. Era and rate-regime cuts"),
        code(
            "print('era cut (split 2016-01-01), bps/yr. SPY/VYM start 2007 (BIL inception),')\n"
            "print('SCHD starts 2011, so its early half is only 4.2 years:')\n"
            "for tk in ('SPY','VYM','SCHD'):\n"
            "    (g1,t1),(g2,t2) = R['era'][tk]\n"
            "    print(f\"  {tk:5s} early {g1:+.2f} (t {t1:+.2f})   late(2016-2026) {g2:+.2f} (t {t2:+.2f})\")\n"
            "print('\\nrate-regime cut (calendar years by realised BIL return, 2% threshold):')\n"
            "for tk in ('SPY','VYM','SCHD'):\n"
            "    (gl,tl,nl),(gh,th,nh) = R['rate'][tk]\n"
            "    print(f\"  {tk:5s} low-rate  {gl:+.2f} (t {tl:+.2f}, {nl} yr)   \"\n"
            "          f\"high-rate {gh:+.2f} (t {th:+.2f}, {nh} yr)\")"
        ),
        md("Positive in **both** halves on all three funds: the sign is era-robust even "
           "though the magnitude never becomes significant.\n\n"
           "The rate-regime cut refutes the naive story. \"DRIP wins most when cash pays "
           "nothing\" is **backwards on this tape** — the high-rate bucket (2023-2026) shows "
           "the *larger* gap, because those were also enormous equity years. The DRIP "
           "advantage is the realised **equity-minus-cash spread** over the delay; the level "
           "of the bill rate says nothing about that spread on its own. (Caveat: the buckets "
           "are unions of non-contiguous calendar years, so the wealth paths inside each are "
           "spliced — read the sign, not the decimals.)"),
        md("## 5. The assumption sweeps\n\n"
           "Three inputs are **not on the tape** and are labelled as assumptions: the pay "
           "lag, the DRIP cost and the sweep cost. Tax is ignored deliberately: both arms "
           "receive the same distribution, of the same size, on the same date, so the bulk "
           "of the tax bill is common. What is *not* common — the sweep arm's bill interest, "
           "taxed as ordinary income, and the two arms' different capital-gain lots — is "
           "second-order here and runs *against* the sweep arm, so ignoring tax understates "
           "DRIP's edge rather than manufacturing it."),
        code(
            "print('pay-lag sweep (ASSUMPTION: Yahoo publishes ex-dates, not pay dates), bps/yr:')\n"
            "print(f\"{'':7s}{'0d':>8s}{'15d':>8s}{'30d*':>8s}{'45d':>8s}   (* = headline)\")\n"
            "for tk in ('SPY','VYM','SCHD'):\n"
            "    print('  %-5s' % tk + ''.join('%8.2f' % x for x in R['lag'][tk]))\n"
            "print('\\ncost sweep (drip_bps, sweep_bps), one-way x amount reinvested, bps/yr:')\n"
            "print(f\"{'':7s}{'(0,0)':>9s}{'(0,2)*':>9s}{'(0,5)':>9s}{'(0,10)':>9s}{'(5,5)':>9s}\")\n"
            "for tk in ('SPY','VYM','SCHD'):\n"
            "    print('  %-5s' % tk + ''.join('%9.2f' % x for x in R['cost'][tk]))"
        ),
        md("**This is the finding that sets the Tradability stamp.** A 0-10 bp cost sweep "
           "moves the gap by under 0.2 bps; the *unobservable* pay lag moves it from +0.7 to "
           "+7.0 bps/yr. When the assumption you had to guess has a wider range than the "
           "quantity you measured, you do not have a measurement."),
        md("## 6. Live synthetic control — the machinery is unbiased\n\n"
           "**Synthetic tape below, not the real tape.** The generator's defaults are a "
           "deliberately loud lab bench (20% premium over cash, 6% distribution yield, a "
           "quiet 6% vol, 20 years) chosen so that an effect worth a few bps a year is "
           "resolvable at all. Planted: DRIP must win. Null (the fund drifts at exactly the "
           "cash rate): the gap must vanish."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from drip_sweep import data, strategy as st\n"
            "for freq in ('Q','A'):\n"
            "    pl = st.seed_sweep(data.synthetic_daily, 1.0, n_seeds=8, sweep_freq=freq)\n"
            "    nl = st.seed_sweep(data.synthetic_daily, 0.0, n_seeds=8, sweep_freq=freq)\n"
            "    print(f\"{freq}: planted {pl['mean']:+7.2f} bps/yr (se {pl['se']:.2f})   \"\n"
            "          f\"null {nl['mean']:+7.2f} bps/yr (se {nl['se']:.2f})\")\n"
            "frames, truth = data.synthetic_panel(signal_strength=1.0, seed=939)\n"
            "print('\\nmonotone in planted distribution yield (quarterly sweep):')\n"
            "for tk, y in zip(truth['tickers'], truth['div_yields']):\n"
            "    g = st.synthetic_detect(frames[tk])['gap_bps_per_year']\n"
            "    print(f\"  yield {y:.0%}: gap {g:+.2f} bps/yr\")"
        ),
        md("## 7. The power calculation that settles the stamp\n\n"
           "**Synthetic tape below.** Same detector, market-realistic parameters: a 5.5% "
           "premium over cash, a 3% distribution yield, 16% volatility, twenty years. If the "
           "detector cannot separate a *true* effect of that size from zero on one path, "
           "then a real-tape *t* of 1.2 is not evidence of absence — it is the expected "
           "reading."),
        code(
            "real = dict(equity_premium=0.055, div_yield_ann=0.03, vol_ann=0.16)\n"
            "pl = st.seed_sweep(data.synthetic_daily, 1.0, n_seeds=8, gen_kw=real)\n"
            "nl = st.seed_sweep(data.synthetic_daily, 0.0, n_seeds=8, gen_kw=real)\n"
            "sep = pl['mean'] - nl['mean']\n"
            "pooled_se = np.sqrt(pl['se']**2 + nl['se']**2)\n"
            "print(f\"planted {pl['mean']:+.2f} bps/yr (sd {pl['sd']:.2f}, se {pl['se']:.2f})\")\n"
            "print(f\"null    {nl['mean']:+.2f} bps/yr (sd {nl['sd']:.2f}, se {nl['se']:.2f})\")\n"
            "print(f\"separation {sep:+.2f} bps/yr against a per-tape sd of ~{pl['sd']:.2f}\")\n"
            "print(f\"-> a SINGLE 20-year tape has |t| ~ {abs(sep)/pl['sd']:.2f} against the null.\")\n"
            "print('   The real tape reads +1.09 to +1.41. That is exactly this.')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The DRIP-minus-sweep gap is **{s['gap']:+.2f} / "
           f"{v['gap']:+.2f} / {c['gap']:+.2f} bps/yr** (SPY / VYM / SCHD), pooled "
           f"**{R['pooled_q']['gap']:+.2f}**, with HAC *t* of {s['t']:+.2f} / {v['t']:+.2f} / "
           f"{c['t']:+.2f} (pooled {R['pooled_q']['t']:+.2f}) and bootstrap CIs that all "
           f"include zero (and the auto-lag *t* is the lowest of the HAC kernels tested, so "
           f"nothing is being hidden by a short window). The effect is correctly signed, "
           f"positive in both eras, ordered by distribution yield — though that ordering "
           f"largely evaporates on matched windows and *reverses* per unit of yield — and "
           f"the synthetic control recovers a planted version at "
           f"*t* ≈ 25 while staying silent on the null "
           f"({R['syn_q_null'][0]:+.2f} ± {R['syn_q_null'][2]:.2f} bps/yr) — but the desk's "
           f"bar is a robust |*t*| ≥ 2 on the **real tape**, and only the annual-sweep "
           f"variant reaches it (SCHD {R['annual']['SCHD'][0]:+.2f} bps/yr, "
           f"*t* = {R['annual']['SCHD'][1]:+.2f}). The power check shows why: at realistic "
           f"parameters one twenty-year path cannot resolve this. **Survivorship** is named — "
           f"three large surviving US ETFs, picked because they are what people own.\n"
           f"- **Tradability — Mirage.** {R['pooled_q']['gap']:+.2f} bps/yr is "
           f"{R['money']['SPY'][1]:.0f}-{R['money']['SCHD'][1]:.0f} currency units a year on "
           f"10,000 — below one ETF ticket's half-spread, invariant to a 0-10 bp cost sweep, "
           f"and swamped by the pay-lag assumption's own +0.7 to +7.0 bps/yr range. Nothing "
           f"here is bankable. The defensible advice is administrative: **DRIP is free and "
           f"removes four decisions a year**; if you must sweep, sweep **quarterly** "
           f"(≈3 bps/yr) rather than **annually** (≈13 bps/yr)."),
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
