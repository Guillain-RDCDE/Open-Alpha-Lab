"""Generate the two narrative notebooks for Study 946 (Distribution is not Return).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Every real-tape number lives in the single frozen ``R`` dict below, which mirrors
docs/results.md. The only live cells run the fast **offline synthetic** control, and they say
so — no synthetic result ever appears under a real-tape banner.
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
# 15 income ETFs + SPY + BIL, monthly, held months 2013-11-30 -> 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2013-11-30", end="2026-06-30", n_months=152,
    # Fingerprints are taken on RETURNS, not levels: auto_adjust=True back-adjusts the whole
    # history on every re-fetch, so a level fingerprint drifts without a return changing.
    fp_tr="ac334d204e26", fp_pr="b94d79082051",
    # the total-return null, annualised (bootstrap CI on the tercile spread)
    hml_lo_ann=-5.67, hml_hi_ann=1.25,
    # the identity: hml_price = hml_total - hml_payout, month-by-month correlation
    ident_corr=0.99995,
    n_funds=15, xs_min=6, xs_max=15,
    # Fama-MacBeth slopes, bps/month per 1 sd of trailing payout rate
    fm_dist=24.6, fm_dist_t=11.27,
    fm_price=-28.1, fm_price_t=-3.84,
    fm_total=-3.5, fm_total_t=-0.48,
    # tercile sort
    dist_hi=9.92, dist_lo=2.87, dist_spread=7.05,
    hml_d=51.6, hml_d_t=10.27, hml_d_lo=42.4, hml_d_hi=60.5,
    hml_p=-69.9, hml_p_t=-4.53, hml_p_lo=-100.5, hml_p_hi=-40.4,
    hml=-18.3, hml_t=-1.24, hml_lo=-47.2, hml_hi=10.4, hml_pgt0=0.102,
    giveback=1.36,
    # excess-of-cash race
    hi_sharpe=0.737, hi_mean=62.6, hi_vol=10.2, hi_dd=-20.1, hi_cagr=9.06,
    lo_sharpe=0.731, lo_mean=80.9, lo_vol=13.3, lo_dd=-23.3, lo_cagr=11.07,
    spy_sharpe=0.862, spy_mean=104.4, spy_vol=14.5, spy_dd=-24.4, spy_cagr=14.02,
    # CAPM
    a_hi=-5.3, t_hi=-0.59, b_hi=0.650, r2_hi=0.859,
    a_lo=-2.5, t_lo=-0.17, b_lo=0.798,
    a_hml=-2.8, t_hml=-0.18, b_hml=-0.148,
    # eras (split 2020-06-30)
    e1_n=80, e1_d=31.2, e1_p=-51.7, e1_pt=-3.28, e1_h=-20.6, e1_ht=-1.31, e1_gb=1.65,
    e2_n=72, e2_d=74.3, e2_p=-90.3, e2_pt=-3.46, e2_h=-15.7, e2_ht=-0.61, e2_gb=1.22,
    # robustness
    q20_p=-67.6, q20_pt=-4.20, q20_h=-8.4, q20_ht=-0.54,
    w40_p=-65.0, w40_pt=-4.54, w40_h=-17.2, w40_ht=-1.23,
    ng_p=-57.0, ng_pt=-2.79, ng_h=-5.4, ng_ht=-0.26,
    dn_p=-68.0, dn_pt=-5.21, dn_h=-20.4, dn_ht=-1.61,
    core_n=74, core_d=58.1, core_p=-50.8, core_pt=-1.93, core_h=7.2, core_ht=0.30,
    # cost x borrow
    to_hi=0.049, to_lo=0.043,
    c0=-18.3, c0_t=-1.24, c5b100=-27.1, c5b100_t=-1.84, c25b200=-37.3, c25b200_t=-2.52,
    # synthetic control
    syn_null_total=0.3, syn_null_total_t=0.10, syn_null_price=-28.8, syn_null_price_t=-8.80,
    syn_null_gb=0.99, syn_plant_total=30.2, syn_plant_total_t=9.20, syn_planted=30.0,
    syn_seeds_mean=-2.4, syn_seeds_sd=3.1,
    # a few per-fund price-only CAGRs (total-return CAGR in brackets in the prose)
    qyld_p=-2.58, qyld_t=8.41, ryld_p=-6.16, ryld_t=5.47, pff_p=-2.55, pff_t=3.69,
    schd_p=9.39, schd_t=12.93, nobl_p=7.94, nobl_t=10.21,
)


HEADER = f"""# Study 946 — Distribution is not Return 💸

**A high-payout ETF advertises a big distribution rate. Does the big number predict a big
return?**

Fifteen listed income funds — QYLD, XYLD, RYLD, JEPI, JEPQ, SPYI, DIVO, NUSI, PBP, PFF and
five dividend-equity funds — are ranked every month on their **trailing-12-month
distribution rate**, reconstructed from the gap between the total-return tape and the
price-only tape. The top third is bought, the bottom third sold, and the pair is held for
the following month (**one execution lag**). We then ask what that rank predicted: the next
payout, the next *price* move, or the next *total* return.

Held months **{R['start']} → {R['end']}** ({R['n_months']} months, cross-section
{R['xs_min']}–{R['xs_max']} funds). Every real number below is frozen from
`docs/results.md` (return fingerprints `{R['fp_tr']}` / `{R['fp_pr']}`); the only live cells
run the **offline synthetic** control and are labelled as such. As-of 2026-06-30.

One thing to keep in view from the first line: the payout is *measured* as the gap between
the two tapes, so "price return" is by definition "total return minus payout". The two
independent facts on this page are how forecastable the **payout** is and how unforecastable
the **total return** is; the price erosion everyone quotes is what those two imply.
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. What a distribution actually is\n\n"
           "When a fund pays you $1, the fund is worth $1 less. The cash does not appear from "
           "nowhere — it comes out of the pot you own a share of. So a \"12% distribution "
           "rate\" is a statement about *how the money is packaged*, not about how much money "
           "there is.\n\n"
           "That gives us a clean test. Every fund has two price series: the **total-return** "
           "one (which pretends you reinvested every payment) and the **price-only** one (the "
           "quote you actually see). The gap between them is exactly what was paid out. So we "
           "can reconstruct each fund's payout rate — and then check what a big one predicts.\n\n"
           "> 🔬 *For the quants:* the reconstruction is `(1+r_total)/(1+r_price) − 1` per "
           "month, compounded over twelve. It lands within a few tenths of a point of every "
           "fund's published sticker, which is the check that the gap really is the "
           "distribution."),
        md("## 2. Look at the two columns side by side\n\n"
           "Before any statistics, just read the tape. The funds with the fattest payouts have "
           "**negative price CAGRs** — their quoted share price has been sinking for a decade. "
           "The funds with the smallest payouts have the fastest-rising prices."),
        code(
            "rows = [\n"
            "    ('QYLD', 11.2, " + repr(R["qyld_p"]) + ", " + repr(R["qyld_t"]) + "),\n"
            "    ('RYLD', 12.5, " + repr(R["ryld_p"]) + ", " + repr(R["ryld_t"]) + "),\n"
            "    ('PFF',   6.5, " + repr(R["pff_p"]) + ", " + repr(R["pff_t"]) + "),\n"
            "    ('SCHD',  3.2, " + repr(R["schd_p"]) + ", " + repr(R["schd_t"]) + "),\n"
            "    ('NOBL',  2.1, " + repr(R["nobl_p"]) + ", " + repr(R["nobl_t"]) + "),\n"
            "]\n"
            "print(f\"{'fund':6s}{'payout':>9s}{'price CAGR':>13s}{'total CAGR':>13s}\")\n"
            "for name, pay, px, tot in rows:\n"
            "    print(f'{name:6s}{pay:8.1f}%{px:12.2f}%{tot:12.2f}%')\n"
            "print()\n"
            "print('The price column sorts itself by payout. The total column does not.')"
        ),
        md("## 3. The three questions, answered\n\n"
           "Rank the whole universe on payout each month and see what the rank forecasts about "
           "the month that follows. Three different things to forecast, three very different "
           "answers."),
        code(
            "R = dict(fm_dist=%r, fm_dist_t=%r, fm_price=%r, fm_price_t=%r,\n"
            "         fm_total=%r, fm_total_t=%r)\n"
            "print('what the payout rank predicts about NEXT MONTH (bps per 1sd of payout):')\n"
            "print('  the next payout      : %%+6.1f bps   t = %%+6.2f   <- almost perfectly forecastable'\n"
            "      %% (R['fm_dist'], R['fm_dist_t']))\n"
            "print('  the PRICE move       : %%+6.1f bps   t = %%+6.2f   <- reliably downward'\n"
            "      %% (R['fm_price'], R['fm_price_t']))\n"
            "print('  the TOTAL return     : %%+6.1f bps   t = %%+6.2f   <- nothing at all'\n"
            "      %% (R['fm_total'], R['fm_total_t']))"
            % (R["fm_dist"], R["fm_dist_t"], R["fm_price"], R["fm_price_t"],
               R["fm_total"], R["fm_total_t"])
        ),
        md("## 4. The give-back ratio\n\n"
           f"Buy the top third by payout and sell the bottom third. The pair collects an extra "
           f"**{R['hml_d']:.1f} bps a month** of distributions — a **{R['dist_spread']:.2f} "
           f"percentage-point** payout spread ({R['dist_hi']:.2f}% versus {R['dist_lo']:.2f}%). "
           f"And it gives back **{-R['hml_p']:.1f} bps a month** in price.\n\n"
           f"That is a **give-back ratio of {R['giveback']:.2f}**. Certainly not zero — the "
           f"free-money reading is dead. Whether it is genuinely *above* one is a different "
           f"question, and the honest answer is that we cannot tell: what is left over — the "
           f"total return — is **{R['hml']:+.1f} bps a month with a *t* of {R['hml_t']:+.2f}**, "
           f"statistically indistinguishable from nothing, and the gap between "
           f"{R['giveback']:.2f} and a clean 1.00 *is* that leftover. Read it as \"one, and no "
           f"evidence of better\".\n\n"
           "> 🔬 *For the quants:* the payout is defined as the total/price gap, so the three "
           "legs obey `price = total − payout` exactly (correlation "
           f"{R['ident_corr']:.5f}). The erosion leg's bootstrap CI is "
           f"[{R['hml_p_lo']:+.1f}, {R['hml_p_hi']:+.1f}] bps and the total leg's is "
           f"[{R['hml_lo']:+.1f}, {R['hml_hi']:+.1f}] — i.e. **[{R['hml_lo_ann']:+.2f}%, "
           f"{R['hml_hi_ann']:+.2f}%] a year**, which rules out the sales pitch and rules out "
           "nothing on the downside."),
        md("## 5. So is the fat-payout basket a bad buy?\n\n"
           f"Not catastrophically — it is just *unremarkable*. Over these "
           f"{R['n_months']} months the high-payout third compounded at "
           f"**{R['hi_cagr']:.2f}%/yr**, the low-payout third at **{R['lo_cagr']:.2f}%**, and "
           f"plain SPY at **{R['spy_cagr']:.2f}%**. Adjusted for how much market risk each one "
           f"carries, the high-payout basket is a **{R['b_hi']:.2f}-beta equity position with "
           f"no alpha** ({R['a_hi']:+.1f} bps/month, *t* = {R['t_hi']:+.2f}).\n\n"
           "You are not being robbed. You are being sold a number that does not mean what the "
           "fact sheet implies it means."),
        md("## 6. Live check — the machinery is honest (offline synthetic)\n\n"
           "**This cell runs a simulation, not the real tape.** We build a fake fund universe "
           "twice. In the first, the payout is pure return of capital: the estimator must find "
           "the erosion and find *nothing* in total return. In the second we secretly plant a "
           "real payout-to-return link: the estimator must find it. If it passes both, the "
           "flat real-tape answer is a fact about income ETFs, not a broken harness."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from dist_illusion import data, strategy as st\n"
            "null,  _     = data.synthetic_panel(signal_strength=0.0, seed=946)\n"
            "plant, truth = data.synthetic_panel(signal_strength=1.0, seed=946)\n"
            "dn, dp = st.synthetic_detect(null), st.synthetic_detect(plant)\n"
            "print('SYNTHETIC (not the real tape)')\n"
            "print('  pure return-of-capital world: total-return slope %+.1f bps (t %+.2f)  '\n"
            "      'price slope %+.1f (t %+.2f)  give-back %.2f'\n"
            "      % (dn['fm_total_bps'], dn['t_total'], dn['fm_price_bps'], dn['t_price'], dn['giveback']))\n"
            "print('  planted-link world         : total-return slope %+.1f bps (t %+.2f)  '\n"
            "      '(we planted %+.1f)'\n"
            "      % (dp['fm_total_bps'], dp['t_total'], truth['planted_slope_per_sd']*1e4))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** \"Distribution is not return\" is not a slogan here, it is the "
           f"tape's arithmetic — and the arithmetic word is doing real work. Two facts are "
           f"measured: the payout rank forecasts the **next payout** (*t* = "
           f"{R['fm_dist_t']:+.2f}) and forecasts **total return not at all** (*t* = "
           f"{R['hml_t']:+.2f}, CI [{R['hml_lo_ann']:+.2f}%, {R['hml_hi_ann']:+.2f}%] a year). "
           f"The **price fall** (*t* = {R['hml_p_t']:+.2f}) then follows by subtraction — it is "
           f"the same evidence in another column, not a third confirmation. Caveats: the "
           f"erosion softens inside the buy-write cohort alone (*t* = {R['core_pt']:+.2f} over "
           f"{R['core_n']} months), the universe only contains funds that survived, and the "
           f"null bounds the sales pitch without ruling out a downside penalty.\n"
           f"- **Tradability — Mirage.** There is nothing to trade. Long low-payout / short "
           f"high-payout earns {-R['hml']:+.1f} bps a month **gross** with *t* = "
           f"{-R['hml_t']:.2f} and a CI straddling zero, and the short leg pays borrow. The "
           f"long-only version trails SPY on return *and* on Sharpe. Use the distribution rate "
           f"as a **transparency tool** — it tells you where the cash is coming from — and "
           f"never as a **signal**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 946 — Distribution is not Return — the teardown\n\n"
           "The Fama-MacBeth cross-sectional slopes, the tercile spread decomposed into payout "
           "and price legs, the give-back ratio, the excess-of-cash race with a CAPM control, "
           "block-bootstrap CIs, the era cut, the sort-width / guard / universe robustness "
           "grid, the cost × borrow sweep, and the live synthetic control.\n\n"
           "Every real-tape number is frozen from `docs/results.md` (fingerprints of the tapes' "
           "**returns** — `%s` / `%s`; level fingerprints are not reproducible because "
           "`auto_adjust=True` back-adjusts the whole history on every re-fetch), "
           "held months %s → %s, n = %d. As-of 2026-06-30.\n\n"
           "**Read the three left-hand sides as two.** The payout is *defined* as the "
           "total/price gap, so `hml_price ≡ hml_total − hml_payout` (correlation %.5f). The "
           "erosion *t* is therefore the payout-persistence *t* carried through a total-return "
           "null — arithmetic, not a second experiment."
           % (R["fp_tr"], R["fp_pr"], R["start"], R["end"], R["n_months"], R["ident_corr"])),
        code("R = %r" % (R,)),
        md("## Design, and the one lag\n\n"
           "Ranking variable: the **trailing-12-month distribution rate**, reconstructed as the "
           "compounded product of `(1+r_total)/(1+r_price) − 1`. It is a **PROXY** for the "
           "marketed sticker — realised trailing rather than last-payment-annualised, and it "
           "includes capital-gains distributions.\n\n"
           "The rank is formed at the close of month *t* and earns month *t+1*'s return. That is "
           "the **only** execution lag in the study. Everything is measured on simple monthly "
           "returns; long legs are excess-of-cash (minus BIL); the self-financing spread pays "
           "borrow (an **ASSUMPTION**, swept)."),
        md("## Fama-MacBeth — three left-hand sides, one right-hand side\n\n"
           "Slopes in bps/month per 1 sd of cross-sectionally z-scored payout rate; "
           "time-series mean over 152 months, Newey-West 6 lags."),
        code(
            "for lab, k, t in [('next payout      ', 'fm_dist', 'fm_dist_t'),\n"
            "                  ('price-only return', 'fm_price', 'fm_price_t'),\n"
            "                  ('TOTAL return     ', 'fm_total', 'fm_total_t')]:\n"
            "    print(f\"{lab}: {R[k]:+7.1f} bps/mo/sd   HAC t = {R[t]:+6.2f}\")"
        ),
        md("## The tercile spread, decomposed\n\n"
           "High minus low, equal-weight, monthly rebalance. The three rows do not just "
           "*happen* to add up — `total = price + payout` is how the payout was measured in the "
           "first place, so exactly two of the three rows carry information and the third is "
           "their difference. The give-back ratio is the same statement again: "
           "`give-back = 1 − hml_total/hml_payout`, so its distance from 1.00 is the "
           "total-return leg, *t* = %.2f." % R["hml_t"]),
        code(
            "print(f\"trailing payout at formation: hi {R['dist_hi']:.2f}%  lo {R['dist_lo']:.2f}%  \"\n"
            "      f\"spread {R['dist_spread']:.2f} pp\")\n"
            "print(f\"payout leg : {R['hml_d']:+7.1f} bps/mo  HAC t {R['hml_d_t']:+6.2f}  \"\n"
            "      f\"boot CI [{R['hml_d_lo']:+.1f}, {R['hml_d_hi']:+.1f}]\")\n"
            "print(f\"price  leg : {R['hml_p']:+7.1f} bps/mo  HAC t {R['hml_p_t']:+6.2f}  \"\n"
            "      f\"boot CI [{R['hml_p_lo']:+.1f}, {R['hml_p_hi']:+.1f}]  <- wholly below zero\")\n"
            "print(f\"TOTAL      : {R['hml']:+7.1f} bps/mo  HAC t {R['hml_t']:+6.2f}  \"\n"
            "      f\"boot CI [{R['hml_lo']:+.1f}, {R['hml_hi']:+.1f}]  P(>0)={R['hml_pgt0']:.3f}\")\n"
            "print(f\"\\ngive-back ratio -price/payout = {R['giveback']:.2f}  \"\n"
            "      f\"(1.00 = a clean wash; the gap from 1.00 IS the total leg, t={R['hml_t']:+.2f})\")\n"
            "print(f\"identity   : hml_price = hml_total - hml_payout -> \"\n"
            "      f\"{R['hml'] - R['hml_d']:+.1f} vs {R['hml_p']:+.1f} bps, corr {R['ident_corr']:.5f}\")\n"
            "print(f\"the TOTAL null in annual terms: [{R['hml_lo_ann']:+.2f}%, {R['hml_hi_ann']:+.2f}%] per year\")"
        ),
        md("> 💡 *In plain words:* the fat payers hand out an extra 52 bps a month and lose an "
           "extra 70 bps of quoted price doing it. Whatever is left over is statistical noise — "
           "but noise with a wide interval: the total-return leg's CI spans "
           f"[{R['hml_lo_ann']:+.2f}%, {R['hml_hi_ann']:+.2f}%] a year, so the tape kills the "
           "marketed *positive* reading and cannot exclude a real negative one."),
        md("## The excess-of-cash race and the CAPM control\n\n"
           "The high-payout cohort is structurally lower-beta (buy-write wrappers cap their "
           "upside), so a raw spread in a bull decade is a beta bet until β is removed."),
        code(
            "print(f\"{'arm':6s}{'exSharpe':>10s}{'mean bps':>10s}{'vol':>8s}{'maxDD':>8s}{'CAGR':>9s}\")\n"
            "for tag, s, m, v, dd, c in [('hi', R['hi_sharpe'], R['hi_mean'], R['hi_vol'], R['hi_dd'], R['hi_cagr']),\n"
            "                            ('lo', R['lo_sharpe'], R['lo_mean'], R['lo_vol'], R['lo_dd'], R['lo_cagr']),\n"
            "                            ('SPY', R['spy_sharpe'], R['spy_mean'], R['spy_vol'], R['spy_dd'], R['spy_cagr'])]:\n"
            "    print(f'{tag:6s}{s:+10.3f}{m:+10.1f}{v:7.1f}%{dd:7.1f}%{c:+8.2f}%')\n"
            "print()\n"
            "print(f\"CAPM hi : alpha {R['a_hi']:+5.1f} bps (t {R['t_hi']:+.2f})  beta {R['b_hi']:+.3f}  R2 {R['r2_hi']:.3f}\")\n"
            "print(f\"CAPM lo : alpha {R['a_lo']:+5.1f} bps (t {R['t_lo']:+.2f})  beta {R['b_lo']:+.3f}\")\n"
            "print(f\"CAPM hml: alpha {R['a_hml']:+5.1f} bps (t {R['t_hml']:+.2f})  beta {R['b_hml']:+.3f}\")"
        ),
        md("## Era cut (split 2020-06-30 — the income-ETF boom line)\n\n"
           "A mechanical identity should hold in **both** halves. It does: the erosion clears "
           "|*t*| = 3 twice, and roughly doubles as the payouts themselves doubled. The "
           "total-return leg is dead in both."),
        code(
            "print(f\"2013-11..2020-06 (n={R['e1_n']}): payout {R['e1_d']:+6.1f}  price {R['e1_p']:+7.1f} \"\n"
            "      f\"(t {R['e1_pt']:+5.2f})  TOTAL {R['e1_h']:+7.1f} (t {R['e1_ht']:+5.2f})  give-back {R['e1_gb']:.2f}\")\n"
            "print(f\"2020-07..2026-06 (n={R['e2_n']}): payout {R['e2_d']:+6.1f}  price {R['e2_p']:+7.1f} \"\n"
            "      f\"(t {R['e2_pt']:+5.2f})  TOTAL {R['e2_h']:+7.1f} (t {R['e2_ht']:+5.2f})  give-back {R['e2_gb']:.2f}\")"
        ),
        md("## Robustness grid — sort width, the corporate-action guard, the universe\n\n"
           "The **guard** is an ASSUMPTION *and a hindsight filter*: fund-months with |total "
           "return| above 0.50 are dropped as unadjusted corporate actions, which means the "
           "filter reads the return of the month being predicted — a fund leaves the sort "
           "formed at *t* because of its *t+1* print. No live trader could run it. Exactly one "
           "fund-month fires — NUSI's 2025-02-18 1-for-2 reverse split, which Yahoo! applied to "
           "neither tape. The **no-guard row is the live-tradable read**, and the panel is also "
           "re-run with NUSI deleted outright (no hindsight anywhere)."),
        code(
            "print(f\"{'variant':26s}{'price (t)':>20s}{'TOTAL (t)':>20s}\")\n"
            "rows = [('tercile 1/3 (headline)', R['hml_p'], R['hml_p_t'], R['hml'], R['hml_t']),\n"
            "        ('quintile-ish 0.20',      R['q20_p'], R['q20_pt'], R['q20_h'], R['q20_ht']),\n"
            "        ('wide sort 0.40',         R['w40_p'], R['w40_pt'], R['w40_h'], R['w40_ht']),\n"
            "        ('no guard (live read)',   R['ng_p'],  R['ng_pt'],  R['ng_h'],  R['ng_ht']),\n"
            "        ('NUSI dropped outright',  R['dn_p'],  R['dn_pt'],  R['dn_h'],  R['dn_ht']),\n"
            "        ('option-income only',     R['core_p'], R['core_pt'], R['core_h'], R['core_ht'])]\n"
            "for n, p, pt, h, ht in rows:\n"
            "    print(f'{n:26s}{p:+13.1f} ({pt:+5.2f}){h:+13.1f} ({ht:+5.2f})')\n"
            "print()\n"
            "print('The erosion survives every cut but the last: inside the nine option-income')\n"
            "print(f\"wrappers alone ({R['core_n']} common months) it softens to t={R['core_pt']:+.2f} and the total\")\n"
            "print('leg turns insignificantly positive. Part of the headline magnitude is')\n"
            "print('cross-cohort (buy-write vs dividend-equity), not a within-cohort law.')"
        ),
        md("## Cost × borrow sweep on the tradable leg\n\n"
           "Turnover is tiny, so friction is not what kills this — the spread is already "
           "insignificant gross. Borrow is an ASSUMPTION (the tape carries none)."),
        code(
            "print(f\"one-way turnover/mo: hi {R['to_hi']:.3f}  lo {R['to_lo']:.3f}\")\n"
            "print(f\"gross              : {R['c0']:+7.1f} bps/mo (t {R['c0_t']:+.2f})\")\n"
            "print(f\"5 bps + 100 bps/yr : {R['c5b100']:+7.1f} bps/mo (t {R['c5b100_t']:+.2f})\")\n"
            "print(f\"25 bps + 200 bps/yr: {R['c25b200']:+7.1f} bps/mo (t {R['c25b200_t']:+.2f})\")\n"
            "print('\\nFriction only deepens a spread that was never positive. There is no')\n"
            "print('cost assumption at which ranking on payout becomes a total-return edge.')"
        ),
        md("## Live synthetic control — **offline simulation, not the real tape**\n\n"
           "Three worlds. The **null**: the payout is pure return of capital — erosion must "
           "fire, total return must not. The **planted** world: a genuine one-for-one "
           "yield-to-return bonus — the total leg must fire and land on the planted value. The "
           "**beta confound**: no alpha, but beta falls across the yield sort — the raw spread "
           "goes negative and the CAPM control must absorb it."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from dist_illusion import data, strategy as st\n"
            "null,  _     = data.synthetic_panel(signal_strength=0.0, seed=946)\n"
            "plant, truth = data.synthetic_panel(signal_strength=1.0, seed=946)\n"
            "conf,  _     = data.synthetic_panel(signal_strength=0.0, beta_slope=0.5, seed=946)\n"
            "dn, dp, dc = st.synthetic_detect(null), st.synthetic_detect(plant), st.synthetic_detect(conf)\n"
            "print('SYNTHETIC (machinery proof, never supports the real-tape stamp)')\n"
            "print('  null    : total %+6.1f (t %+6.2f)  price %+6.1f (t %+6.2f)  give-back %.2f'\n"
            "      % (dn['fm_total_bps'], dn['t_total'], dn['fm_price_bps'], dn['t_price'], dn['giveback']))\n"
            "print('  planted : total %+6.1f (t %+6.2f)  [planted %+6.1f]'\n"
            "      % (dp['fm_total_bps'], dp['t_total'], truth['planted_slope_per_sd']*1e4))\n"
            "print('  beta cfd: raw HML %+6.1f (t %+.2f) -> CAPM alpha %+.1f (t %+.2f), beta %+.3f'\n"
            "      % (dc['hml_bps'], dc['t_hml'], dc['capm_hml']['alpha_bps'],\n"
            "         dc['capm_hml']['t_alpha'], dc['capm_hml']['beta']))\n"
            "seeds = np.array([st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=946+s)[0])['fm_total_bps']\n"
            "                  for s in range(8)])\n"
            "print('  null across 8 seeds: mean %+.1f bps, sd %.1f, |mean|>20 on %d/8'\n"
            "      % (seeds.mean(), seeds.std(ddof=1), (np.abs(seeds) > 20).sum()))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real**, on two measured facts rather than three. (1) The payout rank "
           f"forecasts the **next payout**: *t* = {R['fm_dist_t']:+.2f}, era-stable, and the "
           f"only estimate here that restates nothing else. (2) It is **uninformative about "
           f"total return**: {R['hml']:+.1f} bps/mo, *t* = {R['hml_t']:+.2f}, CI "
           f"[{R['hml_lo_ann']:+.2f}%, {R['hml_hi_ann']:+.2f}%] a year; CAPM α "
           f"{R['a_hml']:+.1f} bps, *t* = {R['t_hml']:+.2f}. The **NAV erosion** "
           f"({R['hml_p']:+.1f} bps/mo, HAC *t* = {R['hml_p_t']:+.2f}, CI "
           f"[{R['hml_p_lo']:+.1f}, {R['hml_p_hi']:+.1f}], |*t*| > 3 in both eras, stable across "
           f"sort widths, guard thresholds and with NUSI deleted) is **the identity of those "
           f"two**, reported as arithmetic and never as independent confirmation. Give-back "
           f"**{R['giveback']:.2f}** — i.e. 1.00 plus an insignificant total leg, so "
           f"more-than-one-for-one erosion is a point estimate, not a finding. Named limits: "
           f"the erosion is partly cross-cohort (*t* = {R['core_pt']:+.2f} within the nine "
           f"option-income wrappers over {R['core_n']} months), the sample is "
           f"**survivorship-selected** (and the sort skips funds without a next-month print), "
           f"the ranking variable is a **PROXY**, and the corporate-action guard is a hindsight "
           f"filter whose no-guard alternative reads {R['ng_p']:+.1f} (*t* = {R['ng_pt']:+.2f}).\n"
           f"- **Tradability — Mirage.** The self-financing expression (high minus low) is "
           f"{R['hml']:+.1f} bps/mo gross (*t* = {R['hml_t']:+.2f}) — i.e. "
           f"{-R['hml']:+.1f} bps if you flip it long-low/short-high — CI straddling zero, and "
           f"the short leg pays borrow it cannot afford. The long-only expression is a "
           f"{R['b_hi']:.2f}-beta clone with α = {R['a_hi']:+.1f} bps (*t* = {R['t_hi']:+.2f}) "
           f"that compounds at {R['hi_cagr']:.2f}%/yr against SPY's {R['spy_cagr']:.2f}%. "
           f"Nothing to bank in either direction."),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    for name, nb in (("01_for_the_curious.ipynb", build_curious()),
                     ("02_for_the_quants.ipynb", build_quants())):
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
