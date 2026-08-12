"""Generate the two narrative notebooks for Study 892 (Corporate-Bond Ladder).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance monthly total
# returns, 8 bond ETFs, joint window 2007-06-30 -> 2026-06-30, 229 months).
R = dict(
    start="2007-06-30", end="2026-06-30", n_months=229, fingerprint="4eeca3d56739",
    ew_dur=7.5, dm_dur=6.0, agg_dur=6.0,
    # duration-matched ladder vs AGG
    lad_ann=3.02, lad_sharpe=0.334, lad_ci=(-0.13, 0.81), lad_dd=-18.3,
    agg_ann=3.07, agg_sharpe=0.386, agg_ci=(-0.08, 0.90), agg_dd=-17.1,
    bnd_ann=3.10, bnd_sharpe=0.396,
    diff_ann=-0.01, diff_bps_mo=-0.11, t_hac=-0.02, t_1s=-0.02,
    diff_sharpe=-0.005, diff_sharpe_ci=(-0.48, 0.46),
    # naive equal-weight ladder
    ew_ann=2.92, ew_sharpe=0.276, ew_dd=-23.2,
    # eras
    era1="2007-2015", era1_diff=0.67, era1_t=0.57,
    era2="2016-2021", era2_diff=-0.60, era2_t=-0.87,
    era3="2022-2026", era3_diff=-0.53, era3_t=-1.09,
    # 2022 shock
    y2021_lad=-2.75, y2021_agg=-1.77, y2021_gap=-0.99,
    y2022_lad=-12.42, y2022_agg=-13.02, y2022_gap=0.60,
    y2023_lad=3.90, y2023_agg=5.66, y2023_gap=-1.75,
    # costs
    cost1=0.9, net1=-0.02, tnet1=-0.04,
    cost2=3.0, net2=-0.04, tnet2=-0.07,
    # synthetic control
    null_t_mean=0.20, null_t_sd=1.36, null_fire=3,
    planted_recovered=1.22, planted_t=3.99,
)


HEADER = f"""# Study 892 — Corporate-Bond Ladder 🪜

**Does a held-to-maturity bond *ladder* beat a constant-maturity bond *fund*?**

The pitch you have heard: a bond **ladder** holds each rung to par and reinvests the cash
at the new yield, while a **fund** is "forced to sell falling bonds" and "locks in losses"
— *"so the ladder shines through a rate shock like 2022."* We race a duration-staggered
Treasury ladder (SHY/IEI/IEF/TLT) against the **AGG**/**BND** funds on total-return closes,
{R['start']} → {R['end']} ({R['n_months']} months), excess of T-bill cash (BIL).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. All eight ETFs are
live survivors; the ~19-year joint window starts at BND/BIL inception.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea — and the catch\n\n"
           "The ladder story *sounds* airtight: hold each bond to maturity and you never "
           "realize a loss. But for a **default-free** bond that is an *accounting* story, "
           "not an *economic* one — the price that fell after rates rose **pulls back to "
           "par** by maturity, and that pull-to-par is the *exact reversal* of the "
           "mark-to-market loss the fund reported. Over a full horizon, two bond portfolios "
           "of the **same duration** earn the **same total return**. So the only real "
           "difference between a ladder and a fund is *how much duration* each carries."),
        code(
            "R = " + repr(R) + "\n"
            "print(f\"duration-matched ladder ({R['dm_dur']}y) vs AGG ({R['agg_dur']}y):\")\n"
            "print(f\"  ladder {R['lad_ann']:+.2f}%/yr, excess-Sharpe {R['lad_sharpe']:.3f}\")\n"
            "print(f\"  AGG    {R['agg_ann']:+.2f}%/yr, excess-Sharpe {R['agg_sharpe']:.3f}\")\n"
            "print(f\"  ladder - fund = {R['diff_ann']:+.2f}%/yr  (HAC t = {R['t_hac']:+.2f})\")\n"
            "print(f\"  difference-Sharpe CI {R['diff_sharpe_ci']} straddles 0\")"
        ),
        md(f"**A statistical dead heat.** Duration-matched, ladder minus fund is "
           f"**{R['diff_ann']:+.2f}%/yr** (HAC *t* = **{R['t_hac']:+.2f}**), and the "
           f"difference-Sharpe confidence interval **{R['diff_sharpe_ci']}** sits right on "
           f"top of zero. No held-to-maturity premium. If anything AGG edges the ladder on "
           f"risk-adjusted terms, thanks to its credit/MBS diversification."),
        md("## 2. The ladder retail actually buys — it *loses*\n\n"
           f"An equal-weight SHY/IEI/IEF/TLT basket has a **{R['ew_dur']}y** duration — "
           f"1.5y longer than AGG, because a quarter of it is 20-year TLT. That extra rate "
           f"risk is not rewarded: excess-Sharpe **{R['ew_sharpe']:.3f}** vs AGG's "
           f"**{R['agg_sharpe']:.3f}**, and a deeper **{R['ew_dd']:.1f}%** drawdown. Every "
           f"apparent 'ladder edge' is a **duration bet in disguise**."),
        code(
            "print(f\"naive equal-weight ladder ({R['ew_dur']}y): {R['ew_ann']:+.2f}%/yr, \"\n"
            "      f\"Sharpe {R['ew_sharpe']:.3f}, maxDD {R['ew_dd']:.1f}%\")\n"
            "print(f\"AGG fund             ({R['agg_dur']}y): {R['agg_ann']:+.2f}%/yr, \"\n"
            "      f\"Sharpe {R['agg_sharpe']:.3f}, maxDD {R['agg_dd']:.1f}%\")"
        ),
        md("## 3. But didn't the ladder win 2022? For one year — then it gave it back\n\n"
           f"In calendar 2022 the pure-Treasury ladder *did* beat AGG by "
           f"**{R['y2022_gap']:+.2f} pp** ({R['y2022_lad']:.2f}% vs {R['y2022_agg']:.2f}%) "
           f"— but not by holding to maturity: it dodged the **credit-spread widening** that "
           f"hit AGG's corporate/MBS sleeve. And it **handed the whole thing back in 2023** "
           f"({R['y2023_gap']:+.2f} pp) as spreads recovered. A one-year composition dodge, "
           f"not a durable edge."),
        code(
            "for yr,lad,agg,gap in [(2021,R['y2021_lad'],R['y2021_agg'],R['y2021_gap']),\n"
            "                       (2022,R['y2022_lad'],R['y2022_agg'],R['y2022_gap']),\n"
            "                       (2023,R['y2023_lad'],R['y2023_agg'],R['y2023_gap'])]:\n"
            "    print(f\"{yr}: ladder {lad:+.2f}%  AGG {agg:+.2f}%  gap {gap:+.2f} pp\")"
        ),
        md("## 4. A live synthetic control — the detector works, the market just says 'no'\n\n"
           "We plant a ladder premium in a seeded toy world and check the detector recovers "
           "it (and stays silent on the null). This is the *machinery* proof — never market "
           "evidence. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from bond_ladder import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(edge_annual=0.0, seed=892))\n"
            "planted = st.synthetic_detect(data.synthetic_world(edge_annual=0.015, seed=892))\n"
            "print('null world   : diff HAC t = %+.2f  (should be ~0)' % null['t_hac'])\n"
            "print('planted +1.5%%: diff HAC t = %+.2f  (should light up)' % planted['t_hac'])"
        ),
        md(f"## The honest verdict\n\n"
           f"- **Signal — None.** Duration-matched, ladder minus fund is "
           f"**{R['diff_ann']:+.2f}%/yr** (HAC *t* = {R['t_hac']:+.2f}), CI on the "
           f"difference-Sharpe {R['diff_sharpe_ci']} straddles zero, and the sign flips era "
           f"to era ({R['era1_diff']:+.2f} / {R['era2_diff']:+.2f} / {R['era3_diff']:+.2f} "
           f"%/yr). The naive ladder underperforms outright. HTM is an accounting illusion "
           f"for default-free bonds.\n"
           f"- **Tradability — Mirage.** No gross edge exists, and the ETF ladder pays "
           f"annual roll costs the one-ticker fund does not — net it is strictly behind. The "
           f"ladder's real appeal is **behavioral** (no realized-loss statements, "
           f"predictable cash flows), not a bankable risk-adjusted return."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 892 — Corporate-Bond Ladder — the teardown\n\n"
           "The excess-of-cash Sharpe race with block-bootstrap CIs, the Newey-West diff "
           "*t*, the era cut, the 2022 calendar-year stress row, the costed net, and the "
           "20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## Headline — duration-matched ladder vs AGG (excess of BIL)"),
        code(
            "print(f\"ladder (dur {R['dm_dur']}y): ann {R['lad_ann']:+.2f}%  exSharpe \"\n"
            "      f\"{R['lad_sharpe']:.3f} CI {R['lad_ci']}  maxDD {R['lad_dd']:.1f}%\")\n"
            "print(f\"AGG    (dur {R['agg_dur']}y): ann {R['agg_ann']:+.2f}%  exSharpe \"\n"
            "      f\"{R['agg_sharpe']:.3f} CI {R['agg_ci']}  maxDD {R['agg_dd']:.1f}%\")\n"
            "print(f\"BND    (dur 5.9y): ann {R['bnd_ann']:+.2f}%  exSharpe {R['bnd_sharpe']:.3f}\")\n"
            "print(f\"ladder - fund : {R['diff_ann']:+.2f}%/yr ({R['diff_bps_mo']:+.2f} bps/mo)  \"\n"
            "      f\"HAC t = {R['t_hac']:+.2f}  1-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"diff-Sharpe   : {R['diff_sharpe']:+.3f}  CI {R['diff_sharpe_ci']} (straddles 0)\")"
        ),
        md("## The naive equal-weight ladder — longer duration, worse Sharpe"),
        code(
            "print(f\"EW ladder (dur {R['ew_dur']}y): ann {R['ew_ann']:+.2f}%  \"\n"
            "      f\"exSharpe {R['ew_sharpe']:.3f}  maxDD {R['ew_dd']:.1f}%  -> LOSES to AGG\")"
        ),
        md("## Era cut — a real premium is stable; this one flips sign, never clears |t|>=1.1"),
        code(
            "for e,d,t in [(R['era1'],R['era1_diff'],R['era1_t']),\n"
            "              (R['era2'],R['era2_diff'],R['era2_t']),\n"
            "              (R['era3'],R['era3_diff'],R['era3_t'])]:\n"
            "    print(f\"{e}: ladder - fund {d:+.2f}%/yr  (HAC t {t:+.2f})\")"
        ),
        md("## 2022 rate shock — a one-year credit-composition dodge, reversed in 2023"),
        code(
            "for yr,lad,agg,gap in [(2021,R['y2021_lad'],R['y2021_agg'],R['y2021_gap']),\n"
            "                       (2022,R['y2022_lad'],R['y2022_agg'],R['y2022_gap']),\n"
            "                       (2023,R['y2023_lad'],R['y2023_agg'],R['y2023_gap'])]:\n"
            "    print(f\"{yr}: ladder {lad:+.2f}%  AGG {agg:+.2f}%  gap {gap:+.2f} pp\")"
        ),
        md("## Tradability — the ladder is rolled annually; the one-ticker fund is free"),
        code(
            "for c,n,t in [(R['cost1'],R['net1'],R['tnet1']),(R['cost2'],R['net2'],R['tnet2'])]:\n"
            "    print(f\"ladder cost {c:.1f} bps/yr -> net diff {n:+.2f}%/yr (HAC t {t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted premium."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from bond_ladder import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_world(edge_annual=0.0, seed=892+s))['t_hac'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_world(edge_annual=0.015, seed=892))\n"
            "print(f\"planted (+1.5%%/yr): recovered {planted['diff_ann_pct']:+.2f}%%/yr, HAC t = {planted['t_hac']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Duration-matched, ladder − fund = "
           f"**{R['diff_ann']:+.2f}%/yr** (HAC *t* = **{R['t_hac']:+.2f}**); the "
           f"difference-Sharpe CI **{R['diff_sharpe_ci']}** straddles zero; the sign flips "
           f"across eras ({R['era1_diff']:+.2f} / {R['era2_diff']:+.2f} / "
           f"{R['era3_diff']:+.2f} %/yr, all |*t*| < 1.1). The naive equal-weight ladder "
           f"underperforms (Sharpe {R['ew_sharpe']:.3f} vs {R['agg_sharpe']:.3f}) purely on "
           f"extra duration. Held-to-maturity vs mark-to-market is an accounting identity "
           f"for default-free bonds; the 20-seed synthetic control recovers a *planted* "
           f"premium at *t* = {R['planted_t']:+.2f}, so the null result is genuine. Short "
           f"~19-year survivor tape.\n"
           f"- **Tradability — Mirage.** No gross edge, and the ETF ladder pays "
           f"{R['cost1']:.1f}–{R['cost2']:.1f} bps/yr of roll cost the buy-and-hold fund "
           f"does not, so net it trails ({R['net1']:+.2f} to {R['net2']:+.2f}%/yr). The "
           f"ladder buys **behavioral comfort**, not risk-adjusted return."),
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
