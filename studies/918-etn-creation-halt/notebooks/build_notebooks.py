"""Generate the two narrative notebooks for Study 918 (Creation Halt).

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


# --------------------------------------------------------------------------- #
# The frozen real-tape headline — mirror of docs/results.md, fingerprint 696cca301c40
# --------------------------------------------------------------------------- #
R = dict(
    asof="2026-06-30", fp="696cca301c40", n_events=6, n_announcement=5,
    # per-event announcement CARs (%) and placebo z at K = 20
    car20={"UNG-2009": -3.30, "USO-2020": -6.00, "VXX-2022": 15.41,
           "OIL-2022": -7.03, "BITO-2021": -0.95},
    z20={"UNG-2009": -0.24, "USO-2020": -1.63, "VXX-2022": 13.84,
         "OIL-2022": -2.55, "BITO-2021": -0.36},
    pooled_z5=-0.91, pooled_t5=-0.25, pos5=3,
    pooled_z10=2.00, pooled_t10=0.61, pos10=3,
    pooled_z20=1.81, pooled_t20=0.60, pos20=1, ci20_lo=-1.90, ci20_hi=7.93,
    # regime drift
    drift={"UNG-2009": (-20.12, -0.60, -11.01), "USO-2020": (-351.48, -2.09, -19.01),
           "VXX-2022": (17.72, 0.71, 19.59), "OIL-2022": (-2.11, -0.25, -2.11),
           "GBTC-2024": (4.35, 0.67, 158.56), "BITO-2021": (-9.09, -0.67, -3.83)},
    # fade
    fade_car={"UNG-2009": -28.29, "USO-2020": 14.58, "VXX-2022": -18.48,
              "OIL-2022": 0.16, "GBTC-2024": -2.64, "BITO-2021": 0.24},
    fade_z={"UNG-2009": -5.02, "USO-2020": 4.25, "VXX-2022": -16.83,
            "OIL-2022": 0.00, "GBTC-2024": -0.31, "BITO-2021": 0.42},
    fade_mean_z=-2.91, fade_t=-0.96, fade_neg=3,
    # robustness
    jack_drop_vxx_z=-1.19, jack_drop_vxx_t=-2.17,
    ruler_exact_z=13.84, ruler_mismatch_z=-1.19, ruler_mismatch_t=-2.17,
    era_early_z=-0.24, era_late_z=2.33, era_late_t=0.60,
    # trade — hindsight exit (held to the resumption date)
    net={"UNG-2009": -12.22, "USO-2020": -19.52, "VXX-2022": 17.80,
         "OIL-2022": -3.78, "GBTC-2024": 126.40, "BITO-2021": -4.80},
    net_mean=17.31, net_median=-4.29, net_pos=2, net_t=0.77,
    net_ex_gbtc_mean=-4.51, net_ex_gbtc_pos=1,
    borrow30_mean=-27.26, borrow30_median=-16.51, borrow10_mean=5.76,
    # trade — BLIND exit (fixed 60 sessions, no resumption date used)
    blind_days=60,
    blind={"UNG-2009": -32.98, "USO-2020": -6.45, "VXX-2022": 0.57,
           "OIL-2022": -2.14, "GBTC-2024": 96.62, "BITO-2021": -8.49},
    blind_mean=7.86, blind_median=-4.29, blind_pos=2, blind_t=0.43,
    # multiplicity
    n_looks=30, fw_bar=0.99833,
    # VXX close-up
    vxx_ann_day=5.81, vxx_in_days=101, vxx_in_bps=17.72, vxx_in_total=19.59,
    vxx_hac_t=0.71, vxx_sharpe=1.10, vxx_ci_lo=-30.3, vxx_ci_hi=69.1, vxx_frac_neg=0.253,
    vxx_car20=15.41, vxx_pct=0.999, vxx_fade=-18.48, vxx_fade_pct=0.000, vxx_nplacebo=1978,
    vxx_pct_indep=0.990, vxx_fade_pct_indep=0.000, vxx_nplacebo_indep=99,
    vxx_path={5: 11.42, 10: 19.10, 20: 15.41, 40: -1.06, 60: 1.79, 101: 17.89},
    # fee drag baked into the spread (bps/day)
    fee={"UNG-2009": -0.54, "USO-2020": -0.01, "VXX-2022": -0.02,
         "OIL-2022": 0.01, "GBTC-2024": 0.79, "BITO-2021": -0.38},
    gbtc_in_net_fee=3.56,
    # synthetic control
    syn_pl_t=2.56, syn_pl_fire=5, syn_pl_bps=10.42, syn_pl_fade=-4.73,
    syn_half_t=0.94, syn_half_fire=3,
    syn_nl_t=-0.73, syn_nl_fire=1, syn_nl_bps=-0.66, syn_nl_fade=0.02,
)


HEADER = f"""# Study 918 — Creation Halt 🚧

**When a fund stops printing new shares, does its price float free — and can you trade it?**

An exchange-traded product is glued to the value of what it holds by exactly one
mechanism: professional dealers create new shares when the price is rich and hand shares
back when it is cheap. Switch the creation side off and the glue only works one way —
supply is frozen, demand is not, and the price should rise to a premium until issuance
restarts. Freeze *redemptions* instead and the same logic runs in reverse, into a
discount.

We test that on **six hardcoded, publicly reported suspensions**: UNG (2009), USO (2020),
VXX and OIL (the same Barclays announcement, 2022), BITO (2021, a capacity constraint —
flagged as the weak one) and GBTC (2015–2024, a redemption freeze). Each capped fund is
measured against an **uncapped** instrument tracking the same thing.

*Real numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fp']}`, as-of {R['asof']}); the only live cells run the offline synthetic control and
say so.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea, and the one honest measuring stick\n\n"
           "To say a fund is 'trading rich' you need something to compare it against. The "
           "funds' own daily net-asset values are not published anywhere free, so we use "
           "the next best thing: an **uncapped** instrument that holds the same thing. For "
           "VXX that is VIXY — a different sponsor's fund tracking the *identical* "
           "VIX-futures index, which was never suspended. For GBTC it is bitcoin itself.\n\n"
           "For the oil and gas funds there is no such twin. The nearest thing sits at a "
           "*different point on the futures curve*, so the comparison mixes the premium we "
           "want with the roll cost we don't. Remember that split — it turns out to be the "
           "whole story.\n\n"
           "> 🔬 **For the quants:** the object is the signed daily log-return spread, "
           "`direction * (Δlog fund − Δlog twin)`, with `direction = −1` for the redemption "
           "freeze so all six events point the same way. It is self-financing, so it is an "
           "excess-of-cash quantity by construction."),
        md("## 2. The one case where you can see it perfectly\n\n"
           "In March 2022 Barclays announced it had issued more VXX notes than it had "
           "registered and stopped selling new ones. VIXY, holding the same index, carried "
           "on as normal. So for five months the market ran a controlled experiment."),
        code(
            "R = dict(vxx_ann_day=%r, vxx_car20=%r, vxx_pct_indep=%r, vxx_nplacebo_indep=%r,\n"
            "         vxx_in_days=%r, vxx_in_total=%r, vxx_fade=%r, vxx_fade_pct_indep=%r,\n"
            "         vxx_path=%r)\n"
            "print('VXX vs VIXY, 2022 suspension (frozen real-tape numbers)')\n"
            "print('  move on the announcement day itself : %%+.2f%%%% (we do NOT claim it - one day of lag)'\n"
            "      %% R['vxx_ann_day'])\n"
            "print('  next 20 sessions, richer than VIXY  : %%+.2f%%%%  (richer than %%.1f%%%% of the %%d independent 20-day windows)'\n"
            "      %% (R['vxx_car20'], R['vxx_pct_indep']*100, R['vxx_nplacebo_indep']))\n"
            "print('  across the whole %%d-session freeze   : %%+.2f%%%%'\n"
            "      %% (R['vxx_in_days'], R['vxx_in_total']))\n"
            "print('  20 sessions after issuance restarts : %%+.2f%%%%  (cheaper than every other window)'\n"
            "      %% R['vxx_fade'])\n"
            "print()\n"
            "print('  but the premium ROUND-TRIPS while the halt is still on:')\n"
            "for k in sorted(R['vxx_path']):\n"
            "    print('    after %%3d sessions: %%+7.2f%%%%' %% (k, R['vxx_path'][k]))"
            % (R["vxx_ann_day"], R["vxx_car20"], R["vxx_pct_indep"], R["vxx_nplacebo_indep"],
               R["vxx_in_days"], R["vxx_in_total"], R["vxx_fade"], R["vxx_fade_pct_indep"],
               R["vxx_path"])
        ),
        md("Out by 15%, back by 18%, on the two dates the theory names. If the story were "
           "going to be true anywhere, this is what it looks like.\n\n"
           "Two cautions we have to put next to it straight away. **First**, the honest "
           "denominator is 99, not 1,978: a 'window' that slides forward one day at a time "
           "overlaps its neighbour by 19 days in 20, so eight and a half years of tape hold "
           "about 99 genuinely independent 20-day windows. **Second**, look at the last "
           "block of that output — the premium is not a steady build. It is up 19% by day "
           "10, back to *minus* 1% by day 40, and up again by the end. Anyone holding it "
           "was on a rollercoaster, not an escalator."),
        md("## 3. And the case that would have taken your arm off\n\n"
           f"April 2020. USO announced it had run out of registered shares the day after "
           f"oil settled at minus $37. Buy the halted fund, short an oil fund that was "
           f"still creating shares — the same trade — and in **six sessions** you were down "
           f"**{R['net']['USO-2020']:.1f}%** net. The fund was not floating up to a premium; "
           f"it was being forced to rebuild its entire portfolio in the middle of the worst "
           f"week the oil market has ever had.\n\n"
           f"That is the problem in one line: the announcement tells you the arbitrage is "
           f"broken. It does not tell you **which way**."),
        md("## 4. Six events, no pattern\n\n"
           f"Line all five dated announcements up and measure the same 20 sessions after "
           f"each (starting the day *after*, so nobody is trading on information they could "
           f"not have had):"),
        code(
            "car20 = %r\n"
            "z20 = %r\n"
            "for k in car20:\n"
            "    verdict = 'as predicted' if z20[k] > 2 else ('opposite' if z20[k] < -2 else 'nothing')\n"
            "    print('  %%-10s %%+7.2f%%%%   (%%s)' %% (k, car20[k], verdict))\n"
            "print('\\npooled across the five: mean standardised move %%+.2f, t = %%+.2f, positive in %%d of 5'\n"
            "      %% (%r, %r, %r))"
            % (R["car20"], R["z20"], R["pooled_z20"], R["pooled_t20"], R["pos20"])
        ),
        md(f"A *t* of **{R['pooled_t20']:+.2f}** is the statistical equivalent of a shrug. "
           f"Take VXX out and the average flips to **{R['jack_drop_vxx_z']:+.2f}** "
           f"(*t* = {R['jack_drop_vxx_t']:+.2f}) — i.e. the remaining halted funds got "
           f"*cheaper*, not richer.\n\n"
           f"> 🔬 **For the quants:** each event's CAR is standardised against that pair's "
           f"own distribution of every other 20-day window (1,000–4,900 controls), so a "
           f"3% move in the gas pair and a 3% move in the VIX pair are not treated as the "
           f"same event."),
        md("## 5. The one thing that does travel: the collapse afterwards\n\n"
           f"The two funds that verifiably *did* carry a premium both handed it back "
           f"violently once the presses restarted — UNG **{R['fade_car']['UNG-2009']:.1f}%** "
           f"and VXX **{R['fade_car']['VXX-2022']:.1f}%** in 20 sessions, both in the extreme "
           f"tail of their own history. That is the mechanism doing exactly what it should. "
           f"It is still only 3 of 6 events pointing the right way "
           f"(*t* = {R['fade_t']:+.2f}), because the other three never had a premium to give "
           f"back in the first place."),
        md("## 6. Why you cannot bank it\n\n"
           f"The trade is: buy the capped fund, sell short the uncapped twin, hold until "
           f"issuance resumes. Commissions are trivial on a five-month hold. **Borrow is "
           f"not** — and the one instrument you must borrow is, by construction, the "
           f"squeezed one.\n\n"
           f"- median result across the six events: **{R['net_median']:.2f}%** net, "
           f"{R['net_pos']} of 6 profitable;\n"
           f"- the average looks positive (**{R['net_mean']:+.2f}%**) only because GBTC's "
           f"8.7-year discount counts as one 'trade'. Drop it and the average is "
           f"**{R['net_ex_gbtc_mean']:+.2f}%**, {R['net_ex_gbtc_pos']} of 5 profitable;\n"
           f"- at a 30%/yr borrow rate — what a genuinely hard-to-borrow capped note costs "
           f"— the average is **{R['borrow30_mean']:+.2f}%**."),
        md("## 6b. And the part that quietly does all the work: knowing when to get out\n\n"
           f"Every number above holds the position **until issuance resumes** — a date "
           f"nobody standing at the announcement could possibly have known. VXX's halt "
           f"lasted 101 sessions; USO's lasted 6. So ask the fair question instead: what "
           f"happens if you just buy on the announcement and sell {R['blind_days']} "
           f"sessions later, like a real person with a calendar and no crystal ball?"),
        code(
            "hind = " + repr(R["net"]) + "\n"
            "blind = " + repr(R["blind"]) + "\n"
            "bd = " + repr(R["blind_days"]) + "\n"
            "print(f\"{'event':<11s}{'told the end':>15s}{('blind ' + str(bd) + 'd'):>15s}\")\n"
            "for k in hind:\n"
            "    print(f'{k:<11s}{hind[k]:>14.2f}%{blind[k]:>14.2f}%')\n"
            "print()\n"
            "print(f\"{'median':<11s}{" + repr(R["net_median"]) + ":>14.2f}%\"\n"
            "      f\"{" + repr(R["blind_median"]) + ":>14.2f}%\")"
        ),
        md(f"Look at the VXX row. Held to the day issuance actually restarted: "
           f"**{R['net']['VXX-2022']:+.2f}%**. Held for a fixed {R['blind_days']} sessions "
           f"because that is all you could have decided in advance: "
           f"**{R['blind']['VXX-2022']:+.2f}%**.\n\n"
           f"The entire profit of the single cleanest example in this study was the exit "
           f"date — and the exit date was hindsight. That, more than the borrow, is why "
           f"this is a Mirage."),
        md("## 7. Live check — is the measuring machine honest? (offline synthetic)\n\n"
           "The cells below are **synthetic**, not the real tape. We build six imaginary "
           "fund/twin pairs, plant a premium that builds during a halt and fades after, and "
           "check the same code finds it — then switch the premium off and check the code "
           "finds nothing."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from creation_halt import strategy as st\n"
            "def avg(ss):\n"
            "    r = [st.synthetic_detect(ss, seed=918 + 11*s) for s in range(8)]\n"
            "    return (np.mean([x['mean_in_bps'] for x in r]),\n"
            "            np.mean([x['mean_fade_z'] for x in r]))\n"
            "pl_bps, pl_fade = avg(1.0)\n"
            "nl_bps, nl_fade = avg(0.0)\n"
            "print('planted halt premium : drift while halted %+.2f bps/day, fade afterwards z %+.2f (should fire)'\n"
            "      % (pl_bps, pl_fade))\n"
            "print('nothing planted      : drift while halted %+.2f bps/day, fade afterwards z %+.2f (should be ~0)'\n"
            "      % (nl_bps, nl_fade))\n"
            "print('(each line is the average of 8 independent synthetic worlds)')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The mechanism is real and you can watch it happen: VXX "
           f"out **{R['vxx_car20']:+.1f}%** and back **{R['vxx_fade']:+.1f}%** against an "
           f"identical uncapped fund, at the extremes of its own history — percentile "
           f"{R['vxx_pct_indep']:.3f} and {R['vxx_fade_pct_indep']:.3f} of the "
           f"{R['vxx_nplacebo_indep']} *independent* 20-day windows it has. We look at "
           f"{R['n_looks']} such comparisons across the study, so neither of those two "
           f"tails would impress on its own; what does is that both happened, in the two "
           f"directions predicted, on the two dates named. But it does not generalise — "
           f"pooled *t* = **{R['pooled_t20']:+.2f}**, {R['pos20']}/5 positive, and USO's "
           f"halt went the other way entirely ({R['net']['USO-2020']:.1f}% net on the same "
           f"trade in six sessions). Our event list is also a *survivor's* list: it "
           f"contains the halts that got reported, and the two most famous (TVIX 2012, the "
           f"original VXX note) are missing because those instruments were delisted and "
           f"their tapes are gone.\n"
           f"- **Tradability — Mirage.** Median net **{R['net_median']:.2f}%**, "
           f"{R['net_pos']}/6 profitable. And the flagship winner evaporates the moment you "
           f"take away the crystal ball: VXX pays **{R['net']['VXX-2022']:+.2f}%** if you "
           f"are told when the halt ends and **{R['blind']['VXX-2022']:+.2f}%** if you are "
           f"not. On top of that, the number that decides the trade — the borrow on a "
           f"squeezed, capped fund — is one nobody publishes."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 918 — Creation Halt — the teardown\n\n"
           "Signed fund-versus-uncapped-twin spreads around six hardcoded issuance "
           "suspensions: placebo-standardised announcement CARs at K = 5/10/20, the "
           "suspended-regime HAC drift with a block-bootstrap CI, the resumption fade, a "
           "leave-one-event-out jackknife, the ruler-quality split, the resumption-date and "
           "cost × borrow sweeps, and the planted/null synthetic control.\n\n"
           "Every real number is frozen from `docs/results.md` (fingerprint `%s`, as-of "
           "%s). One execution lag: announced at `t`, acted at `t+1`, so every window starts "
           "at `t+1` and the announcement-day move is excluded from every statistic.\n\n"
           "> 💡 **In plain words:** we are asking whether a fund that cannot print new "
           "shares drifts richer than an otherwise identical fund that can."
           % (R["fp"], R["asof"])),
        code("R = %r" % (R,)),
        md("## The design\n\n"
           "For each event, `x_t = direction · (Δlog F_t − Δlog P_t)` where `F` is the "
           "capped fund and `P` an uncapped instrument on the same underlying; "
           "`direction = +1` for a creation suspension and `−1` for GBTC's redemption "
           "freeze. `x` is a self-financing long-short spread, so it is an excess-of-cash "
           "quantity by construction. The check `max |raw − excess-of-cash| = 0.00e+00` "
           "against BIL is an **algebraic identity, not a finding** — `(r_f − r_c) − "
           "(r_p − r_c) ≡ r_f − r_p` on any input — and is run only to prove the code path "
           "never races a raw return against an excess one. The economic assumption behind "
           "it (a full cash rebate on the short) is false for a hard-to-borrow name, which "
           "is why borrow is charged separately and swept 0–30 %/yr. Legs are total-return "
           "(`auto_adjust=True`) except `NG=F`, a **price-only** continuous futures print.\n\n"
           "The **ruler** tag matters more than anything else here: *exact* means the twin "
           "holds the same object (VIXY/VXX share an index; BTC-USD is what GBTC holds); "
           "*curve-mismatched* means it sits elsewhere on a futures curve, so the spread "
           "carries roll yield as well as premium.\n\n"
           "Three contaminants are measured rather than waved away: the two legs' "
           "**expense-ratio difference** is inside `x` before any halt effect exists "
           "(`fee_drag_bps`); the regime control **excludes the post-resumption fade**, "
           "which otherwise depresses the baseline and flatters the gap; and the "
           "`hold='halt'` trade **exits on a date nobody knew at entry**, so a blind "
           "fixed-horizon exit is reported beside it."),
        md("## 1. Announcement CARs, standardised by each pair's own placebo distribution"),
        code(
            "for k, (zs, pooled, t, pos) in {\n"
            "    5:  (None, R['pooled_z5'],  R['pooled_t5'],  R['pos5']),\n"
            "    10: (None, R['pooled_z10'], R['pooled_t10'], R['pos10']),\n"
            "    20: (None, R['pooled_z20'], R['pooled_t20'], R['pos20']),\n"
            "}.items():\n"
            "    print(f'K={k:>2}: pooled mean z {pooled:+.2f}  cross-event t {t:+.2f}  positive {pos}/5')\n"
            "print(f\"\\nK=20 event-resample CI on the pooled mean z: [{R['ci20_lo']:+.2f}, {R['ci20_hi']:+.2f}]\")\n"
            "print()\n"
            "for key in R['car20']:\n"
            "    print(f\"  {key:<10s} CAR20 {R['car20'][key]:+7.2f}%   z {R['z20'][key]:+7.2f}\")"
        ),
        md(f"The pooled mean flips sign between K = 5 (−0.91) and K = 10/20 (+2.00/+1.81) "
           f"and the cross-event *t* never exceeds +0.61. With five events and four degrees "
           f"of freedom this is the honest statistic, and it is a null.\n\n"
           f"**Two inference caveats that change how the headline reads.** First, `z` is a "
           f"scale, not a p-value: these spreads are strongly mean-reverting and "
           f"fat-tailed, so VXX's `z = +13.84` sits at empirical percentile "
           f"{R['vxx_pct_indep']:.3f}, not at the 1e−43 a normal table implies. Second, the "
           f"default placebo pool steps one session at a time, so its ~2,000 windows "
           f"overlap by 19/20 — VXX's tape holds **{R['vxx_nplacebo_indep']} independent** "
           f"20-day windows, not 1,978, and `pct_indep` reports on that pool. The design "
           f"inspects **{R['n_looks']} percentiles** (5 events × 3 horizons × 2 legs), so "
           f"the family-wise 5% bar for one look is {R['fw_bar']:.5f} — above the study's "
           f"single best number.\n\n"
           f"> 💡 **In plain words:** across the five halts there is no reliable pattern — "
           f"one of them dominates every number, and even that one is less extreme than the "
           f"raw denominator suggests."),
        md("## 2. Drift while suspended (HAC *t* on the daily signed spread)"),
        code(
            "print(f\"{'event':<11s}{'bps/day':>10s}{'HAC t':>8s}{'total %':>10s}\"\n"
            "      f\"{'fee bps':>10s}{'net of fee':>12s}\")\n"
            "for key, (bps, t, tot) in R['drift'].items():\n"
            "    fee = R['fee'][key]\n"
            "    print(f'{key:<11s}{bps:>10.2f}{t:>8.2f}{tot:>10.2f}{fee:>10.2f}{bps-fee:>12.2f}')\n"
            "print('\\nthe two exact-ruler events (VXX vs VIXY, GBTC vs spot BTC) are the only positive ones')\n"
            "print(f\"...and {R['fee']['GBTC-2024']/R['drift']['GBTC-2024'][0]:.0%} of GBTC's drift is \"\n"
            "      f\"just its 2%/yr fee against an unfeed spot ruler ({R['gbtc_in_net_fee']:+.2f} bps/d net)\")"
        ),
        md("Note the HAC *t*s: even VXX's +19.6% over the freeze is *t* = +0.71 on a daily "
           "basis, and its block-bootstrap CI on the daily mean is "
           f"[{R['vxx_ci_lo']:+.1f}, {R['vxx_ci_hi']:+.1f}] bps/day with "
           f"{R['vxx_frac_neg']:.1%} of resamples negative. The divergence is a level shift "
           "delivered in lumps, not a harvestable daily accrual — which is precisely why "
           "the *event-window* standardisation, not the daily *t*, is the right test here."),
        md("## 3. The resumption fade (K = 20, window starts resume+1)"),
        code(
            "for key in R['fade_car']:\n"
            "    print(f\"  {key:<10s} CAR {R['fade_car'][key]:+7.2f}%   z {R['fade_z'][key]:+7.2f}\")\n"
            "print(f\"\\npooled mean fade z {R['fade_mean_z']:+.2f}  cross-event t {R['fade_t']:+.2f}  \"\n"
            "      f\"negative {R['fade_neg']}/6\")"
        ),
        md("UNG (z = −5.02) and VXX (z = −16.83) are the two funds that demonstrably carried "
           "a premium, and both dump it. The pool is still *t* = −0.96: three of six events "
           "never had a premium to lose.\n\n"
           "> 💡 **In plain words:** the collapse-on-restart half of the story holds "
           "wherever there was something to collapse."),
        md("## 4. Jackknife, ruler split, era cut, date sweep"),
        code(
            "print(f\"drop VXX-2022  -> pooled mean z {R['jack_drop_vxx_z']:+.2f}  t {R['jack_drop_vxx_t']:+.2f}  (sign flips)\")\n"
            "print(f\"ruler = exact             (n=1): mean z {R['ruler_exact_z']:+.2f}\")\n"
            "print(f\"ruler = curve-mismatched  (n=4): mean z {R['ruler_mismatch_z']:+.2f}  t {R['ruler_mismatch_t']:+.2f}  0/4 positive\")\n"
            "print(f\"era cut 2020: early (n=1) z {R['era_early_z']:+.2f} | late (n=4) z {R['era_late_z']:+.2f} (t {R['era_late_t']:+.2f}) -- uninformative by construction\")\n"
            "print('resumption-date sweep +/-10 bd: mean fade z stays in [-5.94, -2.80]; the in-halt drift does NOT survive it')"
        ),
        md("The calendar era cut is honestly useless here (1 event before 2020, 4 after) — "
           "the hardcoded list is too small and too clustered. The cut that *does* carry "
           "information is by ruler quality, and it is damning in a specific way: the four "
           "curve-mismatched pairs are collectively **negative** (*t* = −2.17), which is "
           "not evidence against the mechanism so much as evidence that a roll-mismatched "
           "ruler cannot measure it. Study "
           "[661](../../661-uso-roll-decay/) quantifies exactly the confound at work.\n\n"
           "The resumption-date sweep is the ASSUMPTION check: half the resumption dates "
           "are our public reading rather than a filing date. The *fade* survives ±10 "
           "business days; the in-halt drift number does not (USO's window is six sessions "
           "long), so we do not report the in-halt drift as a result."),
        md("## 5. Tradability — one dollar long the fund, one dollar short the twin\n\n"
           "10 bps one-way × NAV on **both** legs at entry and exit (four crossings), a "
           "**daily rebalancing charge** of `10 bps × Σ|xₜ|` — because `exp(Σx)` is the "
           "return of a continuously dollar-neutral position and holding one flat costs "
           "turnover, 5.73% rather than 0.40% over GBTC's 2,183 sessions — and 3%/yr borrow "
           "on the short leg per calendar day held.\n\n"
           "The `hold='halt'` column exits **on the resumption date**. That is a hindsight "
           "exit: at `t+1` nobody knows the halt will run 101 sessions (VXX) or 6 (USO), "
           "and for the APPROX events the date is our own reading. `hold='blind'` removes "
           "the assumption entirely — enter at `t+1`, exit after a fixed 60 sessions."),
        code(
            "print(f\"{'event':<11s}{'hindsight':>13s}{'blind 60d':>13s}\")\n"
            "for key in R['net']:\n"
            "    print(f\"{key:<11s}{R['net'][key]:>12.2f}%{R['blind'][key]:>12.2f}%\")\n"
            "print(f\"\\n{'mean':<11s}{R['net_mean']:>12.2f}%{R['blind_mean']:>12.2f}%\")\n"
            "print(f\"{'median':<11s}{R['net_median']:>12.2f}%{R['blind_median']:>12.2f}%\")\n"
            "print(f\"{'cross-ev t':<11s}{R['net_t']:>13.2f}{R['blind_t']:>13.2f}\")\n"
            "print(f\"\\nex-GBTC (an 8.7-year regime, not a trade): mean {R['net_ex_gbtc_mean']:+.2f}%  \"\n"
            "      f\"positive {R['net_ex_gbtc_pos']}/5\")\n"
            "print(f\"borrow sensitivity: mean net {R['borrow10_mean']:+.2f}% at 10%/yr, \"\n"
            "      f\"{R['borrow30_mean']:+.2f}% at 30%/yr\")"
        ),
        md(f"**The VXX row is the whole tradability verdict.** Held to the resumption date "
           f"it nets {R['net']['VXX-2022']:+.2f}%; held blind for {R['blind_days']} "
           f"sessions it nets {R['blind']['VXX-2022']:+.2f}%. The premium round-trips to "
           f"{R['vxx_path'][40]:+.2f}% by day 40 and only recovers by the end of the halt, "
           f"so essentially all of the flagship event's P&L was the *timing of the exit* — "
           f"which was hindsight. (GBTC's blind number is not a halt trade at all: 60 "
           f"arbitrary sessions inside an 8.7-year regime.)\n\n"
           f"Commissions move the answer by ~3 pp across a 0–25 bps grid; borrow moves it "
           f"by 50 pp across 0–30%/yr. The borrow rate on a capped, squeezed ETP is not "
           f"published anywhere free, so it is an **ASSUMPTION** and is swept rather than "
           f"chosen — and between the sweep and the blind exit, the verdict is decided."),
        md("## 6. Live synthetic control — the estimator is unbiased and under-powered\n\n"
           "**Synthetic, not the real tape.** Six planted pairs per draw, eight seeds each."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from creation_halt import strategy as st\n"
            "for tag, ss in [('planted', 1.0), ('half', 0.5), ('null', 0.0)]:\n"
            "    res = [st.synthetic_detect(ss, seed=918 + 11*s) for s in range(8)]\n"
            "    t = np.array([r['pooled_t'] for r in res])\n"
            "    b = np.array([r['mean_in_bps'] for r in res])\n"
            "    f = np.array([r['mean_fade_z'] for r in res])\n"
            "    print(f'{tag:<8s} pooled t {t.mean():+.2f}  |t|>=2 in {(abs(t)>=2).sum()}/8  '\n"
            "          f'in-halt {b.mean():+6.2f} bps/d  fade z {f.mean():+.2f}')"
        ),
        md(f"The detector recovers a planted 12 bps/day premium (pooled *t* "
           f"{R['syn_pl_t']:+.2f}, fires {R['syn_pl_fire']}/8), degrades at half strength, "
           f"and is silent on the null (*t* {R['syn_nl_t']:+.2f}, fires {R['syn_nl_fire']}/8). "
           f"Two readings follow. First, the harness is unbiased — the real-tape null is a "
           f"fact about the events, not a broken estimator. Second, **six events buy very "
           f"little power**: even a genuinely large planted premium clears |*t*| ≥ 2 only "
           f"{R['syn_pl_fire']} times in 8. A pooled null on five real events is therefore "
           f"weak evidence of absence, which is one more reason the stamp is Mixed rather "
           f"than None."),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** In the one suspension with an exact uncapped ruler, VXX "
           f"repriced **{R['vxx_car20']:+.2f}%** against VIXY over the 20 lagged sessions "
           f"after the announcement and gave back **{R['vxx_fade']:+.2f}%** after issuance "
           f"resumed. Stated honestly that is percentile {R['vxx_pct_indep']:.3f} and "
           f"{R['vxx_fade_pct_indep']:.3f} of the **{R['vxx_nplacebo_indep']} independent** "
           f"20-day windows in the pair's tape — not 1,978 overlapping ones — so against "
           f"the design's {R['n_looks']} looks (family-wise bar {R['fw_bar']:.5f}) "
           f"**neither tail clears alone**; the *joint* announce-and-fade pattern, in the "
           f"two predicted directions on the two named dates, is what survives, and it is "
           f"n = 1. GBTC, the other exact-ruler case, drifts the predicted way but at "
           f"HAC *t* = +0.67 with {R['fee']['GBTC-2024']/R['drift']['GBTC-2024'][0]:.0%} of "
           f"the drift being nothing but its fee. The pooled result is a null: mean z "
           f"**{R['pooled_z20']:+.2f}**, cross-event *t* **{R['pooled_t20']:+.2f}**, "
           f"{R['pos20']}/5 positive at K = 20, sign-flipping across horizons, and "
           f"*t* = {R['jack_drop_vxx_t']:+.2f} without VXX. No |*t*| ≥ 2 on the pooled "
           f"tape, so not Real; too specific a joint pattern in the one clean pair to call "
           f"None. **Survivorship:** the event list is hand-curated from *reported* "
           f"suspensions and excludes TVIX 2012 and the original VXX note whose tapes did "
           f"not survive delisting — it is biased toward the effect and still cannot pool.\n"
           f"- **Tradability — Mirage.** The flagship winner is an artefact of the exit "
           f"date: VXX nets **{R['net']['VXX-2022']:+.2f}%** held to the resumption "
           f"announcement and **{R['blind']['VXX-2022']:+.2f}%** on a blind "
           f"{R['blind_days']}-session hold, the only rule available in advance. Median "
           f"per-event net **{R['net_median']:.2f}%** at 10 bps / rebalancing / 3%/yr "
           f"borrow, {R['net_pos']}/6 positive, *t* {R['net_t']:+.2f} (hindsight) and "
           f"{R['blind_t']:+.2f} (blind); ex-GBTC mean **{R['net_ex_gbtc_mean']:+.2f}%**, "
           f"{R['net_ex_gbtc_pos']}/5. The decisive input is a borrow rate nobody "
           f"publishes, and at 30%/yr the mean net is **{R['borrow30_mean']:+.2f}%**. Six "
           f"events, half of them with an assumed resumption date, and no way to tell a VXX "
           f"from a USO in advance."),
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
