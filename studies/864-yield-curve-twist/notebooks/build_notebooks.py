"""Generate the two narrative notebooks for Study 864 (Yield-Curve Twist / Butterfly).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily
# total-return, ^FVX/^TNX/^TYX + IEF/TLT/SPY, 2002-07-31 -> 2026-06-30, fp 5136638af800).
R = dict(
    start="2002-07-31", end="2026-06-30", n=6011, fp="5136638af800",
    ief5_t=2.13, ief5_b=3.6, ief21_t=2.34, ief21_b=16.9, ief21_r2=1.58,
    ief63_t=2.42, ief63_b=46.8, ief63_r2=4.04,
    tlt21_t=1.12, tlt21_b=16.1, spy21_t=-1.60, spy21_b=-24.7,
    inc_fly_b=20.2, inc_fly_t=1.98, inc_slope_t=0.25, inc_level_t=-0.92,
    q5=58.5, q1=5.4, q_spread=53.0, q_t=1.66,
    dfly5_t=0.65, dfly21_t=-0.38, dfly63_t=1.56,
    era1_b=48.7, era1_t=3.32, era1_r2=10.32, era1_n=1782,
    era2_b=-1.8, era2_t=-0.18, era2_n=1927,
    era3_b=7.2, era3_t=0.59, era3_n=2050,
    plac_obs_t=2.337, plac_sd=1.047, plac_p=0.024, plac_n=500,
    timer1_active=1.02, timer1_passive=1.41, timer1_spread=-0.39, timer1_t=-0.96,
    timer1_sh_a=0.557, timer1_sh_p=0.522, timer5_spread=-0.61, timer5_sh_a=0.439, sw_yr=13.5,
    plant_t=33.99, plant_spread_t=24.70,
    null_mean_t=0.03, null_sd_t=1.37, null_fire=5, null_seeds=20,
)


HEADER = f"""# Study 864 — Yield-Curve Twist (Butterfly) 🔀

**Beyond level and slope: does the curve's *curvature* predict returns?**

The Treasury curve moves in three modes (Litterman & Scheinkman 1991): a **level** shift, a
**slope** steepening, and a **curvature** — a *butterfly*, the belly moving relative to the two
wings. We build the 5-10-30 butterfly `fly = 2·y10 − y5 − y30` from `^FVX`/`^TNX`/`^TYX` and ask
whether it (and its *change*, a "twist") predicts forward IEF / TLT / SPY returns — *distinct* from
the 2s10s slope (studies 66/132) and roll-down carry (380). Tape: {R['start']} → {R['end']},
{R['n']:,} days (fingerprint `{R['fp']}`).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast synthetic
control. No cross-section → no survivorship bias.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. What is a butterfly?\n\n"
           "Draw a straight line from the **5-year** yield to the **30-year** yield. Where does the "
           "**10-year** (the *belly*) sit relative to that line? The butterfly measures exactly "
           "that gap:\n\n"
           "$$\\text{fly} = 2\\,y_{10} - y_5 - y_{30}$$\n\n"
           "A **positive** fly means the belly yield sits *above* the line — the 10-year is "
           "**cheap** (its price is low). The folklore: a cheap belly mean-reverts, so a high fly "
           "should precede belly bonds (IEF, the 7-10y ETF) *rising*. A *twist* is a day-over-day "
           "**change** in that curvature."),
        code(
            "R = " + repr(R) + "\n"
            "print('butterfly -> forward IEF (belly Treasury) return, HAC t:')\n"
            "print(f\"  5d : beta {R['ief5_b']:+.1f} bps/1sigma   t = {R['ief5_t']:+.2f}\")\n"
            "print(f\"  21d: beta {R['ief21_b']:+.1f} bps/1sigma  t = {R['ief21_t']:+.2f}  (R2 {R['ief21_r2']:.2f}%)\")\n"
            "print(f\"  63d: beta {R['ief63_b']:+.1f} bps/1sigma  t = {R['ief63_t']:+.2f}  (R2 {R['ief63_r2']:.2f}%)\")\n"
            "print('  -> right sign (cheap belly -> belly rallies), full-sample t just past 2.')"
        ),
        md("## 2. But is it stable? The decisive era cut\n\n"
           "A full-sample *t* just past 2 is fragile. Split the tape into three eras and the story "
           "collapses — the **entire** effect is a pre-2010 relic:"),
        code(
            "print('fly -> IEF (h=21), by era:')\n"
            "print(f\"  2002-2009: beta {R['era1_b']:+.1f}  t = {R['era1_t']:+.2f}  (R2 {R['era1_r2']:.1f}%)  <- all the juice\")\n"
            "print(f\"  2010-2017: beta {R['era2_b']:+.1f}  t = {R['era2_t']:+.2f}  <- dead\")\n"
            "print(f\"  2018-2026: beta {R['era3_b']:+.1f}  t = {R['era3_t']:+.2f}  <- dead\")"
        ),
        md("## 3. A live synthetic control — is the machinery honest?\n\n"
           "We plant a butterfly→return edge in a seeded toy world (`fly_signal>0`) and check the "
           "detector recovers it — and stays *silent* on the null (`fly_signal=0`, curvature "
           "wanders but predicts nothing). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from curve_twist import data, strategy as st\n"
            "planted = st.synthetic_detect(data.synthetic_daily(n_days=3000, fly_signal=0.02, seed=864)[0], 21)\n"
            "null    = st.synthetic_detect(data.synthetic_daily(n_days=3000, fly_signal=0.0,  seed=864)[0], 21)\n"
            "print('planted world: regression t = %+.2f  (should light up)' % planted['t'])\n"
            "print('null world   : regression t = %+.2f  (should be ~0)'    % null['t'])"
        ),
        md("## 4. The honest verdict\n\n"
           f"The butterfly **does** predict forward belly-Treasury returns the right way "
           f"(NW *t* = **{R['ief21_t']:+.2f}** at 21d), and it isn't just the 2s10s slope in disguise "
           f"(the slope control is insignificant). **But** the whole effect is a **2002-2009** relic "
           f"(*t* = {R['era1_t']:+.2f}) that dies in 2010-2017 (*t* = {R['era2_t']:+.2f}) and "
           f"2018-2026 (*t* = {R['era3_t']:+.2f}); the *twist* (the change) predicts nothing; and a "
           f"curvature timer **loses to buy-and-hold** after costs. **Signal: Weak** (right sign, "
           f"unstable magnitude), **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 864 — Yield-Curve Twist (Butterfly) — the teardown\n\n"
           "The HAC regression *t*, the incremental slope-control dedup, the Q5−Q1 spread, the "
           "three-era cut, the permutation placebo, the costed timer, and the synthetic control "
           "(with its HAC over-rejection caveat)."),
        code("R = %r" % (R,)),
        md("## The headline — forward log return ~ lagged z(fly), HAC(NW) t"),
        code(
            "print(f\"IEF  5d: beta {R['ief5_b']:+6.1f}  t = {R['ief5_t']:+.2f}\")\n"
            "print(f\"IEF 21d: beta {R['ief21_b']:+6.1f}  t = {R['ief21_t']:+.2f}  R2 {R['ief21_r2']:.2f}%\")\n"
            "print(f\"IEF 63d: beta {R['ief63_b']:+6.1f}  t = {R['ief63_t']:+.2f}  R2 {R['ief63_r2']:.2f}%\")\n"
            "print(f\"TLT 21d: beta {R['tlt21_b']:+6.1f}  t = {R['tlt21_t']:+.2f}   (30y wing is IN the fly -> poor belly proxy)\")\n"
            "print(f\"SPY 21d: beta {R['spy21_b']:+6.1f}  t = {R['spy21_t']:+.2f}   (insignificant, wrong sign for equities)\")"
        ),
        md("## Incremental — is the butterfly just repackaged slope? (dedup)\n\n"
           "Joint fit of forward IEF (h=21) on lagged z-scores of `fly`, `slope` (5s10s) and `level`."),
        code(
            "print(f\"fly  : beta {R['inc_fly_b']:+.1f} bps  t = {R['inc_fly_t']:+.2f}\")\n"
            "print(f\"slope: t = {R['inc_slope_t']:+.2f}  (insignificant -> fly is NOT the slope repackaged)\")\n"
            "print(f\"level: t = {R['inc_level_t']:+.2f}\")\n"
            "print('-> curvature keeps its loading, but its own t falls to 1.98 once level/slope held fixed.')"
        ),
        md("## Quintile spread — Q5(high fly) − Q1(low fly) forward IEF"),
        code(
            "print(f\"h=21: Q5 {R['q5']:+.1f}  Q1 {R['q1']:+.1f}  spread {R['q_spread']:+.1f} bps  t = {R['q_t']:+.2f}  (sub-2)\")"
        ),
        md("## The twist (the *change* in curvature) — nothing"),
        code(
            "print(f\"dfly -> IEF: t = {R['dfly5_t']:+.2f} / {R['dfly21_t']:+.2f} / {R['dfly63_t']:+.2f} at 5/21/63d -> no forward info\")"
        ),
        md("## Robustness — three eras (the spine), fly → IEF h=21"),
        code(
            "print(f\"2002-2009: beta {R['era1_b']:+.1f}  t = {R['era1_t']:+.2f}  R2 {R['era1_r2']:.1f}%  (n={R['era1_n']})  <- all the juice\")\n"
            "print(f\"2010-2017: beta {R['era2_b']:+.1f}  t = {R['era2_t']:+.2f}  (n={R['era2_n']})  <- dead\")\n"
            "print(f\"2018-2026: beta {R['era3_b']:+.1f}  t = {R['era3_t']:+.2f}  (n={R['era3_n']})  <- dead\")"
        ),
        md("## Placebo — shuffle the signal vs forward returns (500 permutations)"),
        code(
            "print(f\"observed |t| = {R['plac_obs_t']:.3f}; permuted sd {R['plac_sd']:.3f} -> two-sided p = {R['plac_p']:.3f}\")\n"
            "print('  (this null destroys the regressor persistence -> an optimistic floor; see the synthetic null below)')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "Own IEF when the lagged fly rank > 0.5, else cash; one-way cost per switch on the NAV."),
        code(
            "print(f\"1 bp: active {R['timer1_active']:+.2f} vs passive {R['timer1_passive']:+.2f} bps -> spread {R['timer1_spread']:+.2f} (t={R['timer1_t']:+.2f}); Sharpe {R['timer1_sh_a']:.3f} vs {R['timer1_sh_p']:.3f}\")\n"
            "print(f\"5 bp: spread {R['timer5_spread']:+.2f} bps/day; Sharpe {R['timer5_sh_a']:.3f}  ({R['sw_yr']:.1f} switches/yr)\")\n"
            "print('-> loses to buy-and-hold on mean return at every cost; the Sharpe sliver is a cash-parking vol artefact.')"
        ),
        md("## Synthetic positive control — unbiased, but a HAC caveat\n\n"
           "Live: the detector must recover a planted edge and (near-)silence on the null. The null "
           "*t* has sd > 1 — the HAC test over-rejects under a persistent regressor + overlapping "
           "returns, so the real full-sample t is discounted."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from curve_twist import data, strategy as st\n"
            "planted = st.synthetic_detect(data.synthetic_daily(n_days=4000, fly_signal=0.02, seed=864)[0], 21)\n"
            "print(f\"planted (fly_signal=0.02): reg t = {planted['t']:+.2f}, Q5-Q1 t = {planted['t_spread']:+.2f}\")\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_daily(n_days=2500, fly_signal=0.0, seed=864+s)[0], 21)['t'] for s in range(8)])\n"
            "print(f\"null (fly_signal=0), 8 seeds: reg t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "print('  sd > 1 -> HAC over-rejects; the real t=2.34 is ~1.8 effective sigma.')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The butterfly predicts forward belly-Treasury (IEF) returns with "
           f"the right sign and a full-sample HAC *t* = **{R['ief21_t']:+.2f}** (β = "
           f"**{R['ief21_b']:+.1f} bps/1σ**), and it is *not* the 2s10s slope repackaged (slope "
           f"control *t* = {R['inc_slope_t']:+.2f}). But it **fails the era bar**: all the juice is "
           f"in 2002-2009 (*t* = {R['era1_t']:+.2f}), dead after (*t* = {R['era2_t']:+.2f} / "
           f"{R['era3_t']:+.2f}); the quintile *t* is {R['q_t']:.2f}, the incremental *t* is "
           f"{R['inc_fly_t']:.2f}, the *twist* is null, and the HAC *t* is inflated (~1.8 effective σ).\n"
           f"- **Tradability — Mirage.** The curvature timer loses to buy-and-hold on mean return at "
           f"every cost ({R['timer1_spread']:+.2f} to {R['timer5_spread']:+.2f} bps/day); the Sharpe "
           f"edge is a cash-parking volatility artefact.\n"
           f"- **Twist beyond level+slope? — Mixed.** Standing curvature holds marginal (unstable) "
           f"info the slope does not; its *change* holds none."),
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
