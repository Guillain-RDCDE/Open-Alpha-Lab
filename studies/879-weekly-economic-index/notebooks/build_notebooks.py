"""Generate the two narrative notebooks for Study 879 (Weekly Economic Index).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (Dallas Fed WEI weekly,
# 2008-01 -> 2026-06, 962 aligned weeks; SPY/XLY/XLP total-return; NW(8) HAC t).
R = dict(
    start="2008-01-12", end="2026-06-13", n=962, fingerprint="94e9e76c22ef",
    spy1_lvl_t=-1.12, spy1_lvl_uni=-0.96, spy1_dwei_t=+1.34, spy1_dwei_uni=+1.16, spy1_r2=0.0033,
    spy4_lvl_t=-1.76, spy4_dwei_t=+0.69, spy4_r2=0.0109,
    rot1_lvl_t=-1.88, rot1_dwei_t=+0.75, rot1_r2=0.0050,
    rot4_lvl_t=-2.24, rot4_lvl_uni=-2.22, rot4_dwei_t=+0.06, rot4_r2=0.0202,
    era_spy1_early_lvl=+0.45, era_spy1_early_dwei=+2.26, era_spy1_late_lvl=-2.69, era_spy1_late_dwei=+0.24,
    era_rot4_early_lvl=-0.59, era_rot4_late_lvl=-2.52,
    cond_spy1_dwei=+0.40, cond_spy1_base=+0.24, cond_spy1_welch=+1.22,
    cond_rot4_wei=+0.05, cond_rot4_base=+0.36, cond_rot4_welch=-1.16,
    placebo_obs_t=2.22, placebo_mean_t=0.80, placebo_p=0.0260,
    ov_wei_gross=-0.11, ov_wei_net=-0.16, ov_dwei_gross=+0.21, ov_dwei_net=+0.02, ov_hold=+0.28,
    ov_dwei_turns=1123,
    null_mean_t=-0.13, null_sd_t=1.12, null_fire=1,
    planted_spy_t=+16.69, planted_rot_t=+22.05,
)


HEADER = f"""# Study 879 — Weekly Economic Index 📅

**Does a *weekly* growth nowcast time the market better than the monthly macro tape?**

The **Weekly Economic Index** (Lewis, Mertens & Stock, 2020) blends **ten weekly** activity
series — Redbook retail, jobless claims, tax withholding, rail traffic, fuel sales,
temp-staffing, steel, electricity, consumer confidence — into one real-time nowcast of U.S.
growth, published every week by the **Dallas Fed**. The claim: its **level** and its
**weekly change** should predict **forward SPY** and the **cyclical-vs-defensive rotation**
(consumer-discretionary `XLY` vs consumer-staples `XLP`). We test it on the real workbook
history ({R['start']} → {R['end']}, {R['n']} aligned weeks).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`);
the live cells run the fast synthetic control. The level uses the revised WEI vintage —
magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Monthly data arrives weeks late. A **weekly** nowcast that reads jobless claims, "
           "retail sales, rail traffic and eight other series *as they land* should, in "
           "principle, let you tilt into cyclicals (`XLY`) and out of defensives (`XLP`) — or "
           "just into stocks — *before* the monthly numbers confirm the turn. Higher "
           "frequency, more timely signal, better timing. That's the theory."),
        code(
            "R = dict(spy1_lvl_t=%r, spy1_dwei_t=%r, rot4_lvl_t=%r, spy1_r2=%r)\n"
            "print('forward SPY (1wk) on WEI level : NW t = %%+.2f  (wrong sign, insignificant)' %% R['spy1_lvl_t'])\n"
            "print('forward SPY (1wk) on weekly chg: NW t = %%+.2f  (right sign, insignificant)' %% R['spy1_dwei_t'])\n"
            "print('XLY-XLP (4wk)   on WEI level  : NW t = %%+.2f  (SIGNIFICANT but WRONG sign)' %% R['rot4_lvl_t'])\n"
            "print('regression R^2 (SPY 1wk)      : %%.4f  (~0.3%%%% of forward variance)' %% R['spy1_r2'])"
            % (R["spy1_lvl_t"], R["spy1_dwei_t"], R["rot4_lvl_t"], R["spy1_r2"])
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`) and check the regression "
           "recovers it — and that it stays *silent* on the null (`edge=0`, a nowcast that "
           "varies but predicts nothing). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from wei import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic(edge=0.0, seed=879, n=700))\n"
            "planted = st.synthetic_detect(data.synthetic(edge=0.010, seed=879, n=700))\n"
            "print('null world   : SPY level t = %+.2f  (should be ~0)' % null['t_level'])\n"
            "print('planted world: SPY level t = %+.2f  (should light up)' % planted['t_level'])"
        ),
        md("## 3. The honest verdict — the nowcast does *not* time the market\n\n"
           f"On the real tape the WEI **level** does not predict forward SPY (NW *t* = "
           f"**{R['spy1_lvl_t']:+.2f}**, and the *wrong* sign), and the **weekly change** — "
           f"the signal that *should* work — is the right sign but insignificant "
           f"(*t* = **{R['spy1_dwei_t']:+.2f}**). Its one significant hit (the weekly change "
           f"predicting SPY at *t* = {R['era_spy1_early_dwei']:+.2f}) lives **entirely in the "
           f"2008–09 recession/recovery** and vanishes after 2017 "
           f"(*t* = {R['era_spy1_late_dwei']:+.2f}). The only overall \\|t\\| ≥ 2 slope — the "
           f"rotation level at 4 weeks (**{R['rot4_lvl_t']:+.2f}**) — is *wrong-signed*: strong "
           f"growth predicts cyclical *under*-performance (a mean-reversion), the opposite of "
           f"the claim. **Signal: None**, **Tradability: Mirage** (no overlay beats "
           f"buy-and-hold). The weekly nowcast is a smooth proxy for the growth cycle, not a "
           f"market-timing edge."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 879 — Weekly Economic Index — the teardown\n\n"
           "The predictive-regression Newey-West (8-lag) HAC *t*, the two-era cut, the "
           "2,000-draw permutation placebo, the costed rotation overlay, and the 20-seed "
           "synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — predictive regression (Newey-West HAC *t*)\n\n"
           "Forward return on a constant + standardized WEI level & weekly change."),
        code(
            "print(f\"SPY 1wk    : R2={R['spy1_r2']:+.4f}  level t={R['spy1_lvl_t']:+.2f}  dwei t={R['spy1_dwei_t']:+.2f}\")\n"
            "print(f\"SPY 4wk    : R2={R['spy4_r2']:+.4f}  level t={R['spy4_lvl_t']:+.2f}  dwei t={R['spy4_dwei_t']:+.2f}\")\n"
            "print(f\"XLY-XLP 1wk: R2={R['rot1_r2']:+.4f}  level t={R['rot1_lvl_t']:+.2f}  dwei t={R['rot1_dwei_t']:+.2f}\")\n"
            "print(f\"XLY-XLP 4wk: R2={R['rot4_r2']:+.4f}  level t={R['rot4_lvl_t']:+.2f}  dwei t={R['rot4_dwei_t']:+.2f}  <- only |t|>=2, WRONG sign\")"
        ),
        md("## Robustness — two eras (split 2017-01-01), univariate HAC *t*\n\n"
           "The one claim-consistent hit (weekly change -> SPY) is an early-era artefact."),
        code(
            "print(f\"SPY 1wk level: early {R['era_spy1_early_lvl']:+.2f}  ->  late {R['era_spy1_late_lvl']:+.2f}\")\n"
            "print(f\"SPY 1wk dwei : early {R['era_spy1_early_dwei']:+.2f}  ->  late {R['era_spy1_late_dwei']:+.2f}   (dies after 2017)\")\n"
            "print(f\"rot 4wk level: early {R['era_rot4_early_lvl']:+.2f}  ->  late {R['era_rot4_late_lvl']:+.2f}   (wrong sign, late-driven)\")"
        ),
        md("## Placebo — permute the nowcast, re-run the HAC slope *t* (rot_h4 level, 2,000 draws)"),
        code(
            "print(f\"observed |t| = {R['placebo_obs_t']:.2f} vs placebo mean |t| = {R['placebo_mean_t']:.2f} -> two-sided p = {R['placebo_p']:.4f}\")\n"
            "print('  the wrong-signed rotation relation is real (mean-reversion) but OPPOSITE to the claim')"
        ),
        md("## The timer — costed long-cyclical / short-defensive overlay, weekly\n\n"
           "One-way 5 bps/leg on turnover + 50 bps/yr borrow on the short; raced vs always-hold."),
        code(
            "print(f\"WEI>median : gross Sh {R['ov_wei_gross']:+.2f}  net Sh {R['ov_wei_net']:+.2f}  vs hold {R['ov_hold']:+.2f}\")\n"
            "print(f\"dwei>0     : gross Sh {R['ov_dwei_gross']:+.2f}  net Sh {R['ov_dwei_net']:+.2f}  vs hold {R['ov_hold']:+.2f}  ({R['ov_dwei_turns']} turns eat it)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null (beyond ~5%) and must recover a planted relation."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from wei import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic(edge=0.0, seed=879+s, n=700))['t_level'] for s in range(10)])\n"
            "print(f\"null (edge=0), 10 seeds: level t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/10\")\n"
            "planted = st.synthetic_detect(data.synthetic(edge=0.010, seed=879, n=700))\n"
            "print(f\"planted (edge=0.010): SPY level t = {planted['t_level']:+.2f}  (recovers cleanly)\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** A weekly growth nowcast does **not** beat the monthly tape. "
           f"The WEI level is insignificant/wrong-signed on forward SPY "
           f"(*t* = {R['spy1_lvl_t']:+.2f}) and era-unstable; the weekly change is the right "
           f"sign but insignificant (*t* = {R['spy1_dwei_t']:+.2f}), its only \\|t\\| ≥ 2 hit a "
           f"2008–09-recovery artefact that dies post-2017 "
           f"({R['era_spy1_early_dwei']:+.2f} → {R['era_spy1_late_dwei']:+.2f}). The single "
           f"overall \\|t\\| ≥ 2 slope — rotation level, {R['rot4_lvl_t']:+.2f} "
           f"(placebo p = {R['placebo_p']:.3f}) — is *wrong-signed* (a mean-reversion). The "
           f"20-seed synthetic control recovers a planted edge (*t* = {R['planted_spy_t']:+.2f}) "
           f"and fires on the null at ~5% ({R['null_fire']}/20), so the null is genuine.\n"
           f"- **Tradability — Mirage.** The costed XLY−XLP overlay under-performs always-hold "
           f"(net Sharpe {R['ov_wei_net']:+.2f} / {R['ov_dwei_net']:+.2f} vs {R['ov_hold']:+.2f}); "
           f"the thin weekly-change edge is eaten by {R['ov_dwei_turns']} weekly turns."),
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
