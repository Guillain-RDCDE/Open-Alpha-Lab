"""Generate the two narrative notebooks for Study 886 (Agency MBS Carry).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return
# closes; monthly duration-neutral carry (MBS-cash) - beta*(IEF-cash); MBB 2007-06 -> 2026-06).
R = dict(
    start="2007-06-30", end="2026-06-30", n_mbb=229, n_vmbs=199, fingerprint="82b3eede7f92",
    mbb_beta=0.521, mbb_r2=0.69, mbb_carry=0.30, mbb_t=0.64, mbb_sharpe=0.13, mbb_dd=-9.4,
    mbb_ci_lo=-0.62, mbb_ci_hi=1.20, mbb_pneg=0.26,
    mbb_static_carry=-0.30, mbb_static_t=-0.45,
    vmbs_beta=0.544, vmbs_carry=0.17, vmbs_t=0.37, vmbs_static_carry=-0.15,
    race_mbs_sh=0.336, race_mbs_mean=1.41, race_mbs_vol=4.20,
    race_ief_sh=0.320, race_ief_mean=2.14, race_ief_vol=6.69, race_adv=0.016, race_welch=-0.40,
    era1_carry=1.83, era1_t=2.50, era2_carry=1.05, era2_t=2.84,
    era3_carry=0.17, era3_t=0.18, hike_carry=1.43, hike_t=3.31,
    lag3_t=0.61, lag6_t=0.64, lag12_t=0.64,
    net_mbb=-0.16, net_mbb_t=-0.33, charge=0.45, net_vmbs=-0.30,
    y2009=8.2, y2022=-4.4,
    syn_null_carry=0.26, syn_null_t=0.67, syn_planted_carry=2.26, syn_planted_t=5.95,
)


HEADER = f"""# Study 886 — Agency MBS Carry 🏠

**Do agency mortgage bonds pay you a real spread over duration-matched Treasuries?**

A mortgage pass-through (MBB, VMBS) is a Treasury bond **plus a short refinancing option**:
homeowners refi when rates fall and sit tight when they rise, so the bond is **negatively
convex** and pays an **option-adjusted spread** as compensation. The carry story says: buy
MBS, duration-hedge with Treasuries (IEF), and pocket that spread. We harvest it as the
duration-neutral, cash-neutral monthly spread on the live ETF tape ({R['start']} → {R['end']}).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`);
the live cells run the fast synthetic control. Short-history caveat: this tape is one rate
cycle plus the 2013 / 2020 / 2022 vol shocks — not many independent draws.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "You lend a homeowner money at a fixed rate. If rates **fall**, he refinances and "
           "hands your money back — right when you'd have wanted to keep earning the old high "
           "rate. If rates **rise**, he keeps the cheap mortgage and you're stuck. Heads you "
           "lose a little, tails you lose a little: that's **negative convexity**, and the "
           "**spread** the mortgage pays over a Treasury is your rent for wearing it. The "
           "carry trade: buy the mortgage bond, short just enough Treasury to cancel the "
           "interest-rate move, and keep the spread."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(mbb_carry=%r, mbb_t=%r, mbb_ci_lo=%r, mbb_ci_hi=%r, race_adv=%r,\n"
            "         race_mbs_sh=%r, race_ief_sh=%r, net_mbb=%r)\n"
            "print('duration-neutral MBS carry (MBB): %%+.2f%%%%/yr  (HAC t = %%+.2f)'\n"
            "      %% (R['mbb_carry'], R['mbb_t']))\n"
            "print('  bootstrap 95%%%% CI: [%%+.2f, %%+.2f] %%%%/yr  -> straddles zero'\n"
            "      %% (R['mbb_ci_lo'], R['mbb_ci_hi']))\n"
            "print('  Sharpe advantage over duration-matched IEF: %%+.3f  (a tie)' %% R['race_adv'])\n"
            "print('  net carry after costs: %%+.2f%%%%/yr  -> below zero' %% R['net_mbb'])"
            % (R["mbb_carry"], R["mbb_t"], R["mbb_ci_lo"], R["mbb_ci_hi"], R["race_adv"],
               R["race_mbs_sh"], R["race_ief_sh"], R["net_mbb"])
        ),
        md("## 2. Where did the spread go?\n\n"
           "The option-adjusted spread is *real* — but you get paid it in calm markets and it "
           "is **clawed straight back** in the rate shocks the convexity exposes you to. The "
           f"one great year was **2009** (+{R['y2009']:.1f}%, post-crash spread compression, a "
           f"one-off); the deep holes are the rate shocks **2008, 2011, and 2022** "
           f"({R['y2022']:.1f}%). Across the full cycle it nets to about **+{R['mbb_carry']:.2f}%/yr** "
           "— indistinguishable from zero, and *negative* once you pay trading costs."),
        md("## 3. Is the machine honest? A live synthetic control\n\n"
           "We plant a known +2%/yr carry in a seeded toy world (a shared rate factor drives "
           "both legs) and check the duration-neutral estimator recovers it — and that it "
           "stays **silent** when there is no carry to find. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from mbs_carry import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(carry_annual=0.0, seed=886))\n"
            "planted = st.synthetic_detect(data.synthetic_world(carry_annual=0.02, seed=886))\n"
            "print('null world   : carry %+.2f%%/yr  HAC t = %+.2f  (should be ~0)'\n"
            "      % (null['carry_ann_pct'], null['t_hac']))\n"
            "print('planted +2%%  : carry %+.2f%%/yr  HAC t = %+.2f  (should light up)'\n"
            "      % (planted['carry_ann_pct'], planted['t_hac']))"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"On the live ETF tape the duration-neutral agency-MBS carry is "
           f"**+{R['mbb_carry']:.2f}%/yr at HAC *t* = +{R['mbb_t']:.2f}** — the right sign, but "
           f"the bootstrap CI **[{R['mbb_ci_lo']:+.2f}, {R['mbb_ci_hi']:+.2f}]** straddles zero, "
           f"the Sharpe edge over duration-matched IEF is **+{R['race_adv']:.3f}** (a tie), the "
           f"carry **collapses to +{R['era3_carry']:.2f}%/yr (*t* +{R['era3_t']:.2f})** in the "
           f"2020-2026 rate-vol era, and the net goes **{R['net_mbb']:+.2f}%/yr after costs**. "
           f"The premium is real ex-ante; negative convexity eats it. **Signal: Weak · "
           f"Tradability: Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 886 — Agency MBS Carry — the teardown\n\n"
           "The duration-neutral carry with empirical vs static-OAD hedges, the HAC *t* and "
           "block-bootstrap CI, the excess-vs-excess Sharpe race, the three-era cut, the "
           "HAC-lag sensitivity, the costed net, and the planted-carry synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — duration-neutral carry `(MBS − cash) − β·(IEF − cash)`\n\n"
           "`β` empirical = realized rate beta (~0.52, well below the static OAD ratio 0.80 — "
           "that gap is the negative-convexity signature)."),
        code(
            "print(f\"MBB empirical  beta {R['mbb_beta']:.3f} (R2 {R['mbb_r2']:.2f}): \"\n"
            "      f\"carry {R['mbb_carry']:+.2f}%/yr  HAC t = {R['mbb_t']:+.2f}  \"\n"
            "      f\"Sharpe {R['mbb_sharpe']:+.2f}  maxDD {R['mbb_dd']:+.1f}%\")\n"
            "print(f\"  bootstrap 95% CI [{R['mbb_ci_lo']:+.2f}, {R['mbb_ci_hi']:+.2f}] %/yr, \"\n"
            "      f\"P(mean<0) = {R['mbb_pneg']:.2f}\")\n"
            "print(f\"MBB static 0.80: carry {R['mbb_static_carry']:+.2f}%/yr  t = {R['mbb_static_t']:+.2f}  \"\n"
            "      f\"-> NEGATIVE under a published-duration hedge\")\n"
            "print(f\"VMBS empirical : carry {R['vmbs_carry']:+.2f}%/yr  t = {R['vmbs_t']:+.2f}  \"\n"
            "      f\"(static {R['vmbs_static_carry']:+.2f}%/yr) -- corroborates\")"
        ),
        md("## Excess-vs-excess Sharpe race — MBS vs duration-matched IEF (both minus cash)"),
        code(
            "print(f\"MBB excess: Sharpe {R['race_mbs_sh']:+.3f} ({R['race_mbs_mean']:+.2f}%/yr, \"\n"
            "      f\"vol {R['race_mbs_vol']:.2f}%)\")\n"
            "print(f\"IEF excess: Sharpe {R['race_ief_sh']:+.3f} ({R['race_ief_mean']:+.2f}%/yr, \"\n"
            "      f\"vol {R['race_ief_vol']:.2f}%)\")\n"
            "print(f\"-> Sharpe advantage {R['race_adv']:+.3f}  (raw Welch t = {R['race_welch']:+.2f}): a tie\")"
        ),
        md("## Robustness — three eras (splits 2014-01, 2020-01)\n\n"
           "The carry clears *t* >= 2 inside the calm sub-eras but dies across 2020-2026 — "
           "the very rate-vol regime the convexity premium exists to compensate for."),
        code(
            "print(f\"2007-2013 (GFC+recovery): {R['era1_carry']:+.2f}%/yr  HAC t = {R['era1_t']:+.2f}\")\n"
            "print(f\"2014-2019 (QE grind)    : {R['era2_carry']:+.2f}%/yr  HAC t = {R['era2_t']:+.2f}\")\n"
            "print(f\"2020-2026 (COVID+hiking): {R['era3_carry']:+.2f}%/yr  HAC t = {R['era3_t']:+.2f}  <- collapses\")\n"
            "print(f\"  (2022+ hiking sub-window: {R['hike_carry']:+.2f}%/yr  t = {R['hike_t']:+.2f}, \"\n"
            "      f\"but the full 2020-26 era it sits in is flat)\")"
        ),
        md("## HAC-lag sensitivity — the thin full-sample *t* is not a lag artefact"),
        code(
            "for lags, t in [(3, R['lag3_t']), (6, R['lag6_t']), (12, R['lag12_t'])]:\n"
            "    print(f\"  NW lags={lags:>2d}: t = {t:+.2f}\")"
        ),
        md("## Tradability — costs push the thin carry below zero\n\n"
           "ETF spreads (MBB 1 bp, IEF 2 bp one-way) on 12 rebalances/yr + 40 bps/yr borrow "
           "on the short Treasury leg."),
        code(
            "print(f\"MBB : gross {R['mbb_carry']:+.2f}%/yr - charge {R['charge']:.2f} \"\n"
            "      f\"-> net {R['net_mbb']:+.2f}%/yr (HAC t = {R['net_mbb_t']:+.2f})\")\n"
            "print(f\"VMBS: net {R['net_vmbs']:+.2f}%/yr\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the estimator must NOT fire on the null and must recover a planted +2%/yr carry."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from mbs_carry import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_world(carry_annual=0.0, seed=886+s))['t_hac'] for s in range(8)])\n"
            "print(f\"null (0 carry), 8 seeds: HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_world(carry_annual=0.02, seed=886))\n"
            "print(f\"planted (+2%%/yr): recovered {planted['carry_ann_pct']:+.2f}%%/yr, HAC t = {planted['t_hac']:+.2f}, beta {planted['beta']:.3f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** Right sign (MBB +{R['mbb_carry']:.2f}%/yr, VMBS "
           f"+{R['vmbs_carry']:.2f}%/yr) and *t* >= 2 inside the calm 2007-13 / 2014-19 sub-eras "
           f"(*t* = +{R['era1_t']:.2f} / +{R['era2_t']:.2f}), but full-sample HAC *t* = "
           f"+{R['mbb_t']:.2f}, bootstrap CI [{R['mbb_ci_lo']:+.2f}, {R['mbb_ci_hi']:+.2f}] "
           f"straddles zero, the carry collapses to +{R['era3_carry']:.2f}%/yr (*t* "
           f"+{R['era3_t']:.2f}) in 2020-2026, flips negative ({R['mbb_static_carry']:+.2f}%/yr) "
           f"under a static-OAD hedge, and the Sharpe advantage over IEF is +{R['race_adv']:.3f}. "
           f"The synthetic control recovers a planted +2%/yr at *t* +{R['syn_planted_t']:.2f} and "
           f"stays flat on the null (*t* +{R['syn_null_t']:.2f}), so this is a genuine *absence* "
           f"of a robust premium, not machinery.\n"
           f"- **Tradability — Mirage.** The ~{R['mbb_carry']:.2f}%/yr gross carry is smaller "
           f"than the ~{R['charge']:.2f}%/yr round-trip friction, so the costed net is "
           f"**{R['net_mbb']:+.2f}%/yr (MBB) / {R['net_vmbs']:+.2f}%/yr (VMBS)**; a proper "
           f"static-duration hedge turns the spread negative before any cost. Negative convexity "
           f"eats the option-adjusted spread."),
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
