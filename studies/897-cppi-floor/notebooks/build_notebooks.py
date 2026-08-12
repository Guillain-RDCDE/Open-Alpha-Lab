"""Generate the two narrative notebooks for Study 897 (CPPI Floor).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the frozen
``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic control, so
execution is quick and network-free. Every code cell that references ``R`` is a plain runtime
f-string; ``R`` itself is injected once, in its own cell.
"""

from __future__ import annotations

import os
import pprint

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers — mirror of docs/results.md (SPY risky, BIL cash, total return,
# 2007-05-31 -> 2026-06-30; CPPI m=5, floor 80% accreting at cash, long-only).
R = dict(
    start="2007-05-31", end="2026-06-30", n_days=4801, fp="4b5a30d250b4",
    mult=5, floor=80,
    cppi_sharpe=0.164, cppi_cagr=2.37, cppi_vol=7.94, cppi_dd=-20.8,
    bh_sharpe=0.542, bh_cagr=10.70, bh_vol=19.78, bh_dd=-55.2,
    static_sharpe=0.542, static_cagr=5.27, static_vol=7.42, static_dd=-23.0,
    sharpe_vs_bh=-0.379, alpha_ann=-1.56, t_alpha=-1.16, beta=0.705, diff_t_nw=-2.15,
    avg_w=0.38, turnover=2.3, n_breach=0,
    boot_point=-0.379, boot_lo=-0.691, boot_hi=-0.050, boot_win=1.4,
    crash08_bh=-36.8, crash08_bh_dd=-47.1, crash08_c=-11.6, crash08_c_dd=-11.2,
    crash20_bh=18.3, crash20_bh_dd=-33.7, crash20_c=-4.8, crash20_c_dd=-16.3,
    crash22_bh=-18.2, crash22_bh_dd=-24.5, crash22_c=-18.7, crash22_c_dd=-20.8,
    era_e_cppi=-0.464, era_e_bh=0.335, era_e_vs=-0.799, era_e_t=-2.38, era_e_n=2164,
    era_l_cppi=0.751, era_l_bh=0.759, era_l_vs=-0.008, era_l_t=-0.74, era_l_n=2637,
    m3_sh=0.517, m3_dd=-33.5, m3_w=0.67, m4_sh=0.417, m4_dd=-31.4,
    m5_sh=0.164, m5_dd=-20.8, m6_sh=-0.242, m6_dd=-18.0, m8_sh=-0.288, m8_dd=-19.2,
    fl70_sh=0.323, fl70_dd=-31.9, fl70_cagr=4.83, fl80_sh=0.164, fl80_dd=-20.8, fl80_cagr=2.37,
    fl90_sh=0.123, fl90_dd=-10.6, fl90_cagr=1.78,
    cost0=-0.379, cost1=-0.394, cost2=-0.413, cost5=-0.472, cost10=-0.554, cost10_dd=-17.3,
    gap_thresh=20, gap_breach_at=20,
    bear_prot=0.520, bear_prot_min=0.110, bear_breach=0, calm_prot=0.005,
    gap25_fire=20, gap15_fire=0, n_seeds=30,
)

R_CELL = "R = " + pprint.pformat(R, width=100)

BOOT = ("import os, sys\n"
        "sys.path.insert(0, os.path.abspath('..'))\n"
        "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n")


HEADER = f"""# Study 897 — CPPI Floor 🧱

**Build a trapdoor-proof floor by rule alone: hold a fixed multiple of your "cushion" in stocks, the
rest in bills, and de-risk automatically as the cushion shrinks. Does the floor really hold — and what
does it cost?**

**Constant-Proportion Portfolio Insurance** (Black & Jones 1987; Perold & Sharpe 1988) promises a
mechanical downside floor with no options: set a protected floor `F` (here **{R['floor']}%** of NAV,
accreting at the cash rate), and each day hold `m ×` the cushion `(V − F)` in SPY — multiplier
**m = {R['mult']}** — and the rest in BIL bills. As the market falls the cushion shrinks and the rule
sells stock; if it hits zero you are cash-locked on the floor. Real tape SPY/BIL,
{R['start']} → {R['end']}.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fp']}`); the live cells run
the fast synthetic control. Short history: BIL lists 2007 — a single, GFC-anchored cycle (which is
exactly where a floor should shine).*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        code(R_CELL),
        md("## 1. The floor really holds — that part is true\n\n"
           "The rule is purely mechanical and it does exactly what it says on the tin: it caps the "
           "drawdown. Over 2007–2026 CPPI's worst peak-to-trough was a **third** of buy-and-hold's, "
           "and the floor was **never breached**."),
        code(
            "print(f\"CPPI (m={R['mult']}, floor {R['floor']}%): maxDD {R['cppi_dd']:.1f}%   \"\n"
            "      f\"floor breaches {R['n_breach']}\")\n"
            "print(f\"buy-and-hold SPY        : maxDD {R['bh_dd']:.1f}%\")\n"
            "print(f\"2008 crash: buy-and-hold {R['crash08_bh']:+.1f}% (DD {R['crash08_bh_dd']:.1f}%)\"\n"
            "      f\"  ->  CPPI {R['crash08_c']:+.1f}% (DD {R['crash08_c_dd']:.1f}%)\")"
        ),
        md("## 2. …but it is paid for in full, out of the upside\n\n"
           "Insurance is never free. The floor is bought by holding a lot of cash, so CPPI compounds "
           f"at **{R['cppi_cagr']:.1f}%/yr** while buy-and-hold does **{R['bh_cagr']:.1f}%** — and its "
           f"risk-adjusted return (excess-of-cash **Sharpe {R['cppi_sharpe']:.2f}**) is a *third* of "
           f"buy-and-hold's (**{R['bh_sharpe']:.2f}**). The bootstrap is emphatic: the Sharpe shortfall "
           f"is real with ~99% confidence."),
        code(
            "print(f\"excess-Sharpe:  CPPI {R['cppi_sharpe']:.3f}  vs  buy-and-hold {R['bh_sharpe']:.3f}\")\n"
            "print(f\"CAGR:           CPPI {R['cppi_cagr']:.2f}%   vs  buy-and-hold {R['bh_cagr']:.2f}%\")\n"
            "print(f\"bootstrap Sharpe gain (CPPI - BH): {R['boot_point']:+.3f}  \"\n"
            "      f\"95% CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]  P(CPPI wins) {R['boot_win']:.1f}%\")"
        ),
        md("## 3. The 2020 trap — mechanical insurance whipsaws on a V\n\n"
           "A floor tuned for a *crash* has a nasty failure mode: it de-risks into a fast drop and is "
           "still in cash for the snap-back. In 2020 CPPI sold into the Feb–March crater and missed the "
           f"recovery — turning buy-and-hold's **+{R['crash20_bh']:.1f}%** year into **{R['crash20_c']:+.1f}%**. "
           "And 2022's slow grind barely tripped the cushion, so it was hardly helped at all."),
        code(
            "for yr,bh,c in [(2020,R['crash20_bh'],R['crash20_c']),(2022,R['crash22_bh'],R['crash22_c'])]:\n"
            "    print(f\"{yr}: buy-and-hold {bh:+.1f}%   ->   CPPI {c:+.1f}%\")"
        ),
        md("## 4. A live synthetic control — the engine is honest\n\n"
           "We plant a **bear** world (the floor must visibly hold) and a **calm** null (nothing to "
           "protect), and check the machinery reads each correctly. No network."),
        code(
            BOOT +
            "from cppi import data, strategy as st\n"
            "bear = st.synthetic_detect(data.synthetic_prices(seed=897, n_days=2500, drift=-0.0004, sigma=0.016)[0])\n"
            "calm = st.synthetic_detect(data.synthetic_prices(seed=897, n_days=2500, drift=0.0004, sigma=0.006)[0])\n"
            "print(f\"bear world: drawdown protection {bear['dd_protection']:+.3f}, floor breaches {bear['n_breach']}\")\n"
            "print(f\"calm null : drawdown protection {calm['dd_protection']:+.3f}  (nothing to protect)\")"
        ),
        md("## 5. The honest verdict\n\n"
           f"- **Signal: Mixed.** The floor-protection claim is **real and mechanical** (maxDD "
           f"{R['cppi_dd']:.1f}% vs {R['bh_dd']:.1f}%, zero breaches, 2008 cut from −47% to −11%). But "
           f"the implied *'insurance improves the risk-adjusted outcome'* claim is **false**: CPPI's "
           f"Sharpe is {R['cppi_sharpe']:.2f} vs {R['bh_sharpe']:.2f} (CI clear of zero), and its dynamic "
           f"re-timing *subtracts* value ({R['alpha_ann']:+.2f}%/yr). The floor is real; the free lunch "
           f"is not.\n"
           f"- **Tradability: Mirage.** As a way to *make money* there is nothing to bank — CPPI trails "
           f"buy-and-hold gross, before a single basis point of cost. The drawdown cap is cheap and real "
           f"**as insurance**, but insurance is a cost (2.4% CAGR vs 10.7%), not a paycheck — and an "
           f"overnight gap past 1/m = {R['gap_thresh']}% would breach even the floor."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 897 — CPPI Floor — the teardown\n\n"
           "The three-book race (CPPI / buy-and-hold / matched static), the excess-of-cash identity, "
           "the leverage-clean re-timing spanning alpha, the block-bootstrap Sharpe-difference CI, the "
           "two-era cut, the multiplier / floor / cost sweeps, the gap-risk stress, and the 30-seed "
           "synthetic control."),
        code(R_CELL),
        md("## The race — CPPI vs buy-and-hold vs matched static mix (excess-of-cash, gross)\n\n"
           "A *constant* fraction of SPY funded from cash has the **same** excess-of-cash Sharpe as SPY "
           "— so the matched-static and buy-and-hold Sharpes coincide, and CPPI's only distinguishing "
           "act is the dynamic **re-timing** (the spanning alpha)."),
        code(
            "print(f\"CPPI        : excess-Sharpe {R['cppi_sharpe']:.3f}  CAGR {R['cppi_cagr']:5.2f}%  \"\n"
            "      f\"vol {R['cppi_vol']:5.2f}%  maxDD {R['cppi_dd']:.1f}%\")\n"
            "print(f\"buy-and-hold: excess-Sharpe {R['bh_sharpe']:.3f}  CAGR {R['bh_cagr']:5.2f}%  \"\n"
            "      f\"vol {R['bh_vol']:5.2f}%  maxDD {R['bh_dd']:.1f}%\")\n"
            "print(f\"static@avgw : excess-Sharpe {R['static_sharpe']:.3f}  CAGR {R['static_cagr']:5.2f}%  \"\n"
            "      f\"vol {R['static_vol']:5.2f}%  maxDD {R['static_dd']:.1f}%\")\n"
            "print(f\"vs buy-and-hold {R['sharpe_vs_bh']:+.3f}   re-timing alpha {R['alpha_ann']:+.2f}%/yr \"\n"
            "      f\"(HAC t {R['t_alpha']:+.2f}, beta {R['beta']:.3f})   avg-w {R['avg_w']:.2f}, \"\n"
            "      f\"turnover {R['turnover']:.1f}x/yr, breaches {R['n_breach']}\")"
        ),
        md("## Bootstrap — circular block CI on the excess-Sharpe difference (CPPI − buy-and-hold)"),
        code(
            "print(f\"gain {R['boot_point']:+.3f}  95% CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]  \"\n"
            "      f\"P(CPPI wins) {R['boot_win']:.1f}%   -> the CI sits entirely below zero\")"
        ),
        md("## Crash years — the floor's best day (2008) and worst (2020)"),
        code(
            "for yr,bh,bd,c,cd in [(2008,R['crash08_bh'],R['crash08_bh_dd'],R['crash08_c'],R['crash08_c_dd']),\n"
            "                      (2020,R['crash20_bh'],R['crash20_bh_dd'],R['crash20_c'],R['crash20_c_dd']),\n"
            "                      (2022,R['crash22_bh'],R['crash22_bh_dd'],R['crash22_c'],R['crash22_c_dd'])]:\n"
            "    print(f\"{yr}: BH {bh:+6.1f}% (DD {bd:6.1f}%)  ->  CPPI {c:+6.1f}% (DD {cd:6.1f}%)\")"
        ),
        md("## Robustness — two eras (split 2016-01-01)\n\n"
           "The Sharpe shortfall concentrates in the volatile first half (re-timing alpha significantly "
           "negative); the calmer half ties on Sharpe but the drawdown protection *fades* (a "
           "cash-accreting floor no longer sits under an appreciated NAV — a TIPP/ratchet floor fixes this)."),
        code(
            "print(f\"2007-2015 (n={R['era_e_n']}): CPPI-Sh {R['era_e_cppi']:+.3f}  BH-Sh {R['era_e_bh']:+.3f}  \"\n"
            "      f\"vs-BH {R['era_e_vs']:+.3f}  alpha-t {R['era_e_t']:+.2f}\")\n"
            "print(f\"2016-2026 (n={R['era_l_n']}): CPPI-Sh {R['era_l_cppi']:+.3f}  BH-Sh {R['era_l_bh']:+.3f}  \"\n"
            "      f\"vs-BH {R['era_l_vs']:+.3f}  alpha-t {R['era_l_t']:+.2f}\")"
        ),
        md("## The frontier — multiplier & floor sweeps (no setting beats buy-and-hold's 0.542)"),
        code(
            "print('multiplier:')\n"
            "for m,sh,dd in [(3,R['m3_sh'],R['m3_dd']),(4,R['m4_sh'],R['m4_dd']),(5,R['m5_sh'],R['m5_dd']),\n"
            "                (6,R['m6_sh'],R['m6_dd']),(8,R['m8_sh'],R['m8_dd'])]:\n"
            "    print(f\"  m={m} (breach>{100//m}%): CPPI-Sh {sh:+.3f}  maxDD {dd:.1f}%\")\n"
            "print('floor:')\n"
            "for fl,sh,dd,cg in [(70,R['fl70_sh'],R['fl70_dd'],R['fl70_cagr']),(80,R['fl80_sh'],R['fl80_dd'],R['fl80_cagr']),\n"
            "                    (90,R['fl90_sh'],R['fl90_dd'],R['fl90_cagr'])]:\n"
            "    print(f\"  floor {fl}%: CPPI-Sh {sh:+.3f}  maxDD {dd:.1f}%  CAGR {cg:.2f}%\")"
        ),
        md("## Costed & gap risk — no return edge to erode, and the floor isn't a hard guarantee\n\n"
           "CPPI trails buy-and-hold **at zero cost** (not a friction story); and a one-day drop past "
           "**1/m = 20%** breaches the floor — daily rebalancing can't pre-empt an overnight gap."),
        code(
            "for c,vs in [('0bp',R['cost0']),('1bp',R['cost1']),('2bp',R['cost2']),('5bp',R['cost5']),('10bp',R['cost10'])]:\n"
            "    print(f\"  {c:>4}: CPPI vs buy-and-hold {vs:+.3f}\")\n"
            "print(f\"gap risk: a single overnight drop > 1/m = {R['gap_thresh']}% breaches the floor \"\n"
            "      f\"(SPY never gapped that far close-to-close in 2007-2026, so the real-tape floor held)\")"
        ),
        md("## Synthetic control — the machinery is unbiased (live, offline)"),
        code(
            BOOT +
            "import numpy as np\n"
            "from cppi import data, strategy as st\n"
            "bear = np.array([st.synthetic_detect(data.synthetic_prices(seed=897+s, n_days=2500, drift=-0.0004, sigma=0.016)[0])['dd_protection'] for s in range(8)])\n"
            "calm = np.array([st.synthetic_detect(data.synthetic_prices(seed=897+s, n_days=2500, drift=0.0004, sigma=0.006)[0])['dd_protection'] for s in range(8)])\n"
            "gap  = np.array([st.synthetic_detect(data.synthetic_prices(seed=897+s, n_days=4000, drift=0.0003, sigma=0.011, gap_prob=0.01, gap_size=0.25)[0])['n_breach'] for s in range(8)])\n"
            "print(f\"bear (8 seeds): drawdown protection mean {bear.mean():+.3f}  (floor holds)\")\n"
            "print(f\"calm (8 seeds): drawdown protection mean {calm.mean():+.3f}  (nothing to protect)\")\n"
            "print(f\"gap 25%>20% (8 seeds): floor breach fired in {(gap>0).sum()}/8\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — MIXED.** The floor-protection claim is **real and mechanical**: maxDD "
           f"**{R['cppi_dd']:.1f}%** vs buy-and-hold's **{R['bh_dd']:.1f}%**, **{R['n_breach']}** breaches, "
           f"2008 cut from −47% to −11%, confirmed by a 30-seed synthetic control. But the implied "
           f"*'insurance improves the risk-adjusted outcome'* claim is **false**: excess Sharpe "
           f"**{R['cppi_sharpe']:.3f} vs {R['bh_sharpe']:.3f}** (bootstrap gain {R['boot_point']:+.3f}, "
           f"95% CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]), no multiplier or floor beats buy-and-hold, "
           f"and the re-timing spanning alpha **subtracts** value ({R['alpha_ann']:+.2f}%/yr, *t* = "
           f"{R['era_e_t']:+.2f} in the volatile era). A single-cycle, GFC-anchored, ~19-year window.\n"
           f"- **Tradability — MIRAGE.** No bankable *return* edge — CPPI trails buy-and-hold gross, "
           f"before costs. The drawdown cap is cheap and genuine **as insurance** (2.3×/yr turnover on "
           f"SPY, survives every cost) but it is a *cost* paid in full out of upside "
           f"({R['cppi_cagr']:.1f}% CAGR vs {R['bh_cagr']:.1f}%), not a paycheck — and an overnight gap "
           f"past 1/m = {R['gap_thresh']}% breaches even the floor."),
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
