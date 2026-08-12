"""Generate the two narrative notebooks for Study 888 (CLO AAA Carry).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# JAAA/ICLO/LQD/IEF/BKLN/BIL, 2020-01-02 -> 2026-06-30; excess-of-cash = minus BIL).
R = dict(
    asof="2026-06-30", fp="3568185876b2", n_rows=1631,
    # per-leg excess-of-cash race (Sharpe-sorted)
    jaaa_start="2020-10-20", jaaa_n=1429, jaaa_exc=1.38, jaaa_vol=1.63, jaaa_sharpe=0.84,
    jaaa_lo=0.09, jaaa_hi=1.72, jaaa_t=2.33, jaaa_dd=-2.60, jaaa_fracneg=0.015, jaaa_bpsday=0.55,
    iclo_start="2022-12-12", iclo_n=889, iclo_exc=2.14, iclo_vol=2.43, iclo_sharpe=0.88,
    iclo_lo=0.22, iclo_hi=2.31, iclo_t=3.08, iclo_dd=-3.46,
    bkln_exc=1.82, bkln_vol=7.71, bkln_sharpe=0.24, bkln_t=0.53, bkln_dd=-24.17,
    lqd_exc=-1.09, lqd_vol=10.27, lqd_sharpe=-0.11, lqd_t=-0.27, lqd_dd=-24.95,
    ief_exc=-2.42, ief_vol=7.46, ief_sharpe=-0.33, ief_t=-0.90, ief_dd=-23.92,
    # head-to-heads (JAAA - bench)
    h2h_lqd=4.10, h2h_lqd_t=1.23, h2h_ief=5.85, h2h_ief_t=2.00, h2h_bkln=-0.86, h2h_bkln_t=-0.60,
    # era cut
    zirp_n=427, zirp_exc=0.00, zirp_sharpe=0.00, zirp_t=0.00,
    high_n=875, high_exc=1.84, high_sharpe=1.37, high_t=2.96,
    # tradability
    cost1_net=1.35, cost1_sharpe=0.83, cost1_t=2.28,
    cost12_net=1.02, cost12_sharpe=0.62, cost12_t=1.72,
    rel_gross=4.10, rel_charge=1.12, rel_net=2.98, rel_sharpe=0.35, rel_t=0.89,
    # synthetic control
    null_sharpe_mean=0.03, null_sharpe_sd=0.34, null_fire=0,
    planted_exc=1.77, planted_sharpe=1.73, planted_t=3.90, planted_lo=0.86, planted_hi=2.58,
)


HEADER = f"""# Study 888 — CLO AAA Carry 🔒

**Does the senior slice of a CLO pay a *real* pickup over cash and same-rated corporates?**

A **AAA-rated CLO tranche** sits at the top of a collateralised-loan-obligation waterfall:
first to be paid, last to take losses, protected by a thick cushion of subordinate tranches.
It pays a spread over cash and over same-rated IG corporate bonds — the story goes — as
compensation for *structural complexity* (a securitisation few desks underwrite), not for
credit risk (realized senior-tranche defaults are ~nil). Since 2020-10 it trades in a liquid,
floating-rate ETF (**JAAA**; **ICLO** from 2022-12), so it carries almost no *duration* either.

We test that carry **excess-of-cash** (minus BIL) against IG corporates (**LQD**), Treasuries
(**IEF**), and — the sharp control — the *un-tranched* leveraged-loan collateral itself
(**BKLN**). {R['jaaa_start']} → {R['asof']}.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. **Short history:** JAAA ~5.7y, ICLO ~3.5y — ONE rate cycle, NO CLO
credit-stress event in-sample, so realized Sharpes are an upper bound (named on the Signal
axis). This is distinct from **[614-clo-equity-yield](../../614-clo-equity-yield/)**, the
risky first-loss BOTTOM of the same stack.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one line\n\n"
           "Buy the *safest* slice of a pool of leveraged loans — the AAA tranche, shielded by "
           "everything below it — and you still get paid a spread over T-bills for the sheer "
           "hassle of the structure. Because the tranche floats, there's almost no interest-rate "
           "risk; because it's senior, there's almost no default risk. The question is whether "
           "that 'complexity spread' is a *real* risk-adjusted edge or just repackaged cash."),
        code(
            "R = " + repr(R) + "\n"
            "print(f\"JAAA excess-of-cash : {R['jaaa_exc']:+.2f}%/yr on {R['jaaa_vol']:.2f}% vol \"\n"
            "      f\"-> Sharpe {R['jaaa_sharpe']:+.2f}  (95% CI [{R['jaaa_lo']:+.2f}, {R['jaaa_hi']:+.2f}], HAC t {R['jaaa_t']:+.2f})\")\n"
            "print(f\"  its worst drawdown in ~5.7y: {R['jaaa_dd']:+.2f}%  (a floating senior tranche barely wobbles)\")\n"
            "print(f\"ICLO (2nd AAA-CLO fund): Sharpe {R['iclo_sharpe']:+.2f}  (HAC t {R['iclo_t']:+.2f}) -- an independent confirm\")"
        ),
        md("## 2. The tell — the AAA tranche vs the *raw loans* it's built from\n\n"
           "The cleanest control isn't corporate bonds — it's **BKLN**, the same leveraged loans "
           "held *un-tranched* and below investment grade. If seniority/tranching is doing real "
           "work, the AAA slice should deliver a far better *risk-adjusted* return than the whole."),
        code(
            "print(f\"JAAA  : Sharpe {R['jaaa_sharpe']:+.2f}   maxDD {R['jaaa_dd']:+6.2f}%   (senior AAA tranche)\")\n"
            "print(f\"BKLN  : Sharpe {R['bkln_sharpe']:+.2f}   maxDD {R['bkln_dd']:+6.2f}%   (un-tranched leveraged loans)\")\n"
            "print(f\"LQD   : Sharpe {R['lqd_sharpe']:+.2f}   maxDD {R['lqd_dd']:+6.2f}%   (IG corporates, ~8y duration)\")\n"
            "print(f\"IEF   : Sharpe {R['ief_sharpe']:+.2f}   maxDD {R['ief_dd']:+6.2f}%   (7-10y Treasuries)\")\n"
            "print('--> the AAA slice earns MORE per unit of risk than the loans it is carved from,')\n"
            "print('    and dodged the -24% drawdowns that duration (LQD/IEF) and raw loans (BKLN) took.')"
        ),
        md("## 3. Is the detector honest? A live synthetic control\n\n"
           "We plant a steady +1.2%/yr excess-of-cash carry in a seeded toy world (plus a high-vol "
           "'duration' decoy that earns *nothing* extra) and check the detector recovers it — and "
           "stays silent on the null (`carry=0`). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from clo_aaa import data, strategy as st\n"
            "planted = data.synthetic_world(carry_annual=0.012, seed=888).rename(columns={'cash':'BIL','carry':'CARRY','dur':'DUR'})\n"
            "nullw   = data.synthetic_world(carry_annual=0.0,   seed=888).rename(columns={'cash':'BIL','carry':'CARRY','dur':'DUR'})\n"
            "p = st.carry_stats(planted, 'CARRY', cash='BIL', n_boot=400)\n"
            "z = st.carry_stats(nullw,   'CARRY', cash='BIL', n_boot=400)\n"
            "print(f\"planted carry: excess {p['excess_ann_pct']:+.2f}%/yr  Sharpe {p['sharpe']:+.2f}  HAC t {p['t_hac']:+.2f}  (should light up)\")\n"
            "print(f\"null world   : excess {z['excess_ann_pct']:+.2f}%/yr  Sharpe {z['sharpe']:+.2f}  HAC t {z['t_hac']:+.2f}  (should be ~0)\")"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"**Signal — Real (but thin & regime-bound).** The AAA-CLO carry *is* a genuine "
           f"risk-adjusted pickup: JAAA earns **{R['jaaa_exc']:+.2f}%/yr** over cash on just "
           f"**{R['jaaa_vol']:.2f}%** vol (**Sharpe {R['jaaa_sharpe']:+.2f}**, HAC *t* "
           f"{R['jaaa_t']:+.2f}, bootstrap CI clear of zero), dominating IG corporates "
           f"(Sharpe {R['lqd_sharpe']:+.2f}), Treasuries ({R['ief_sharpe']:+.2f}) *and* the "
           f"un-tranched loan collateral ({R['bkln_sharpe']:+.2f}) — so the seniority earns real "
           f"keep. ICLO independently confirms ({R['iclo_sharpe']:+.2f}). **But** it's a "
           f"regime bet: in the ZIRP era the excess was **{R['zirp_exc']:+.2f}%/yr** (nothing to "
           f"pocket when cash is 0); all of it is the {R['high_exc']:+.2f}%/yr high-rate era. And "
           f"the ~5.7y sample has **no CLO stress event** — the one AAA-CLO mark-down (Mar-2020) "
           f"predates JAAA.\n\n"
           f"**Tradability — Fragile.** It survives costs trivially (buy-and-hold, the ~0.20% ER "
           f"already in the NAV, tight spreads → net **{R['cost1_net']:+.2f}%/yr**) and capacity "
           f"is ample (JAAA is a >$20bn fund). What makes it *Fragile* not *Investable*: the whole "
           f"~1.4%/yr IS an insurance premium for a tail (illiquidity/credit) the calm sample never "
           f"charged — a single crisis mark-down (AAA CLOs fell ~5-10% in Mar-2020) would erase "
           f"years of carry, so the realized Sharpe flatters the true edge."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 888 — CLO AAA Carry — the teardown\n\n"
           "The excess-of-cash Sharpe race (block-bootstrap CIs), the HAC *t* on the daily "
           "excess, the JAAA-vs-benchmark head-to-heads, the ZIRP-vs-high-rate era cut, the "
           "costed harvest & relative trade, and the seeded synthetic control."),
        code("R = " + repr(R)),
        md("## The race — excess-of-cash (minus BIL) Sharpe, full available history per leg\n\n"
           "Annualised excess, vol, Sharpe [95% block-bootstrap CI], HAC *t* on the daily excess, "
           "max drawdown. Sorted by Sharpe."),
        code(
            "rows = [('ICLO',R['iclo_exc'],R['iclo_vol'],R['iclo_sharpe'],R['iclo_lo'],R['iclo_hi'],R['iclo_t'],R['iclo_dd']),\n"
            "        ('JAAA',R['jaaa_exc'],R['jaaa_vol'],R['jaaa_sharpe'],R['jaaa_lo'],R['jaaa_hi'],R['jaaa_t'],R['jaaa_dd']),\n"
            "        ('BKLN',R['bkln_exc'],R['bkln_vol'],R['bkln_sharpe'],None,None,R['bkln_t'],R['bkln_dd']),\n"
            "        ('LQD', R['lqd_exc'], R['lqd_vol'], R['lqd_sharpe'], None,None,R['lqd_t'], R['lqd_dd']),\n"
            "        ('IEF', R['ief_exc'], R['ief_vol'], R['ief_sharpe'], None,None,R['ief_t'], R['ief_dd'])]\n"
            "print(f\"{'leg':<5}{'exc%/yr':>9}{'vol%':>7}{'Sharpe':>8}{'  95% CI':>16}{'HACt':>7}{'maxDD%':>9}\")\n"
            "for leg,ex,vol,sh,lo,hi,t,dd in rows:\n"
            "    ci = f\"[{lo:+.2f},{hi:+.2f}]\" if lo is not None else '        -       '\n"
            "    print(f\"{leg:<5}{ex:>+9.2f}{vol:>7.2f}{sh:>+8.2f}{ci:>16}{t:>+7.2f}{dd:>+9.2f}\")"
        ),
        md("The AAA-CLO funds (JAAA, ICLO) sit at the **top** on Sharpe with **tiny** vol and "
           "shallow drawdowns; the duration/credit alternatives (LQD, IEF) have *negative* "
           "excess Sharpe over a window that contained the 2022 bond crash, and the un-tranched "
           "loans (BKLN) earn a similar raw excess but at ~5x the vol and a -24% drawdown."),
        md("## Head-to-head — JAAA excess minus each benchmark excess (== JAAA − bench)"),
        code(
            "for b,d,t in [('LQD',R['h2h_lqd'],R['h2h_lqd_t']),('IEF',R['h2h_ief'],R['h2h_ief_t']),\n"
            "              ('BKLN',R['h2h_bkln'],R['h2h_bkln_t'])]:\n"
            "    print(f\"JAAA - {b:<4}: {d:+6.2f}%/yr  HAC t {t:+.2f}\")\n"
            "print('  (JAAA beats duration handily; vs BKLN it is ~flat in RAW return -- but at a fraction of the risk)')"
        ),
        md("## Era cut — ZIRP (≤2022-06, rates ~0) vs the high-rate plateau (2023+)\n\n"
           "The carry is **regime-dependent**: an excess-of-cash spread is proportionally tiny "
           "when the base rate is zero, and AAA-CLO spreads widened through 2022H1."),
        code(
            "print(f\"ZIRP   (n={R['zirp_n']}): JAAA excess {R['zirp_exc']:+.2f}%/yr  Sharpe {R['zirp_sharpe']:+.2f}  HAC t {R['zirp_t']:+.2f}\")\n"
            "print(f\"HighRt (n={R['high_n']}): JAAA excess {R['high_exc']:+.2f}%/yr  Sharpe {R['high_sharpe']:+.2f}  HAC t {R['high_t']:+.2f}\")\n"
            "print('  --> essentially ALL the carry is the high-rate era; the ZIRP era is flat (not negative).')"
        ),
        md("## Tradability — does a costed net edge survive?\n\n"
           "(a) Buy-and-hold JAAA funded by cash; the ~0.20%/yr ER is ALREADY inside the "
           "total-return NAV, so the extra friction is only the ETF bid-ask on rebalances "
           "(3 bps one-way × turnover). (b) The relative isolation trade long JAAA / short LQD "
           "pays borrow + spread on both legs — and is also short ~8y duration (a rate bet)."),
        code(
            "print(f\"(a)  1 rebal/yr : net {R['cost1_net']:+.2f}%/yr  Sharpe {R['cost1_sharpe']:+.2f}  HAC t {R['cost1_t']:+.2f}\")\n"
            "print(f\"(a) 12 rebal/yr : net {R['cost12_net']:+.2f}%/yr  Sharpe {R['cost12_sharpe']:+.2f}  HAC t {R['cost12_t']:+.2f}  (needless churn)\")\n"
            "print(f\"(b) long JAAA/short LQD: gross {R['rel_gross']:+.2f}% - charge {R['rel_charge']:.2f}% -> net {R['rel_net']:+.2f}%/yr  Sharpe {R['rel_sharpe']:+.2f}  HAC t {R['rel_t']:+.2f}\")\n"
            "print('  --> the harvest survives costs easily; costs are NOT the binding constraint. The tail risk is.')"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: a planted carry must be recovered; the null (`carry=0`) must NOT fire."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from clo_aaa import data, strategy as st\n"
            "def W(c, s): return data.synthetic_world(carry_annual=c, seed=s).rename(columns={'cash':'BIL','carry':'CARRY','dur':'DUR'})\n"
            "null_t = np.array([st.carry_stats(W(0.0, 888+s), 'CARRY', cash='BIL', n_boot=200)['t_hac'] for s in range(8)])\n"
            "print(f\"null (carry=0), 8 seeds: HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "p = st.carry_stats(W(0.012, 888), 'CARRY', cash='BIL', n_boot=400)\n"
            "print(f\"planted (+1.2%/yr): excess {p['excess_ann_pct']:+.2f}%/yr  Sharpe {p['sharpe']:+.2f}  HAC t {p['t_hac']:+.2f}  CI [{p['sharpe_lo']:+.2f},{p['sharpe_hi']:+.2f}]\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real (thin & regime-bound).** JAAA's excess-of-cash carry is "
           f"**{R['jaaa_exc']:+.2f}%/yr** on **{R['jaaa_vol']:.2f}%** vol — **Sharpe "
           f"{R['jaaa_sharpe']:+.2f}** (HAC *t* {R['jaaa_t']:+.2f}, bootstrap CI "
           f"[{R['jaaa_lo']:+.2f}, {R['jaaa_hi']:+.2f}] clear of zero), the **top** of the "
           f"excess-vs-excess Sharpe race and *above the un-tranched loans it's built from* "
           f"(BKLN {R['bkln_sharpe']:+.2f}), with ICLO confirming ({R['iclo_sharpe']:+.2f}, *t* "
           f"{R['iclo_t']:+.2f}). Caveats named loudly: the carry lives in the high-rate era "
           f"(ZIRP flat, {R['zirp_exc']:+.2f}%/yr), the CI only *just* clears zero, and ~5.7y "
           f"spans a single stress-free cycle.\n"
           f"- **Tradability — Fragile.** The harvest survives costs and capacity easily "
           f"(net **{R['cost1_net']:+.2f}%/yr**, Sharpe {R['cost1_sharpe']:+.2f}), but it is thin "
           f"and its realized Sharpe is flattered by a sample with no CLO stress: the ~1.4%/yr IS "
           f"the premium for a tail that never fired, so it is real-but-fragile, not bankable free "
           f"money. The synthetic control fires on {R['null_fire']}/many nulls — the engine is honest."),
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
