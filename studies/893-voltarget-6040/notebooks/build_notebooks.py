"""Generate the two narrative notebooks for Study 893 (Vol-Target 60/40).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the frozen
``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic control, so
execution is quick and network-free. Every code cell that references ``R`` is a plain runtime
f-string (single ``%`` literals); ``R`` itself is injected once, in its own cell.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (SPY/IEF 60/40, BIL cash,
# total return, 2007-05-31 -> 2026-06-30, 21-day window, matched-risk target 10.67%/yr).
R = dict(
    start="2007-05-31", end="2026-06-30", n_days=4801, fp="a874e54fa109",
    target_vol=10.67,
    static_sharpe=0.682, static_cagr=8.43, static_vol=10.68, static_dd=-29.6,
    vt_sharpe=0.808, vt_cagr=10.33, vt_vol=11.26, vt_dd=-24.2,
    sharpe_gain=0.126, alpha_ann=2.31, t_alpha=1.92, beta=0.931, diff_t_nw=1.63,
    avg_lev=1.36, frac_lev=74, frac_capped=18, turnover=9.4,
    boot_point=0.126, boot_lo=-0.086, boot_hi=0.323, boot_win=87.8,
    era_e_gain=0.217, era_e_t=1.86, era_e_n=1891, era_e_dd_s=-29.6, era_e_dd_v=-25.6,
    era_l_gain=0.058, era_l_t=0.93, era_l_n=2868, era_l_dd_s=-21.4, era_l_dd_v=-16.4,
    win21=0.126, win42=0.088, win63=0.054,
    cost0=0.107, cost1=0.098, cost2=0.090, cost5=0.065, cost10=0.023, cost10_dd=-25.3,
    crash08_s=-14.9, crash08_v=-13.2, crash22_s=-17.0, crash22_v=-14.1,
    null_t_mean=0.20, null_fire=2, plan_t_mean=3.09, plan_fire=23, n_seeds=30,
)

R_CELL = "R = " + pprint.pformat(R, width=100)

BOOT = ("import os, sys\n"
        "sys.path.insert(0, os.path.abspath('..'))\n"
        "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n")


HEADER = f"""# Study 893 — Vol-Target 60/40 🌡️

**Put a thermostat on the balanced book: hold *less* when it's stormy, *more* when it's calm — does
the classic 60/40 come out better?**

The 60/40 (60% stocks, 40% bonds) is the sensible default ([Study 97](../../97-balancing-act/) grades
it Real & Investable). Its risk, though, is *not* constant — it doubles in a crisis. So we bolt on the
inverse-volatility overlay this desk certified on equities ([Study 16, Storm-Shy](../../16-storm-shy/)):
scale the *whole book's* exposure so realized **portfolio** volatility stays near a constant target.
Real tape SPY/IEF, cash = BIL, {R['start']} → {R['end']}.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fp']}`); the live cells run
the fast synthetic control. Short history: BIL lists 2007 — a single-cycle, GFC-anchored window.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        code(R_CELL),
        md("## 1. The thermostat, in one line\n\n"
           "Volatility **clusters** — a wild week is usually followed by another wild week, a calm one "
           "by calm. So *yesterday's* realized vol is a decent guess for *today's*. The rule uses only "
           "the past: `weight = target_vol / recent_portfolio_vol`, capped at 2×. When the 60/40 gets "
           "stormy you automatically hold less; when it's sleepy you hold a bit more (financed at "
           "cash). Same average risk, just **re-timed**."),
        code(
            "print(f\"static 60/40   : excess Sharpe {R['static_sharpe']:.3f}, maxDD {R['static_dd']:.1f}%\")\n"
            "print(f\"vol-targeted   : excess Sharpe {R['vt_sharpe']:.3f}, maxDD {R['vt_dd']:.1f}%\")\n"
            "print(f\"Sharpe gain    : {R['sharpe_gain']:+.3f}  (drawdown cut by {R['static_dd']-R['vt_dd']:+.1f} pts)\")"
        ),
        md("## 2. The honest catch — a smoother ride, but the Sharpe lift is *not* certain\n\n"
           f"The point estimates all lean the thermostat's way (Sharpe **{R['static_sharpe']:.2f} → "
           f"{R['vt_sharpe']:.2f}**, drawdown **{R['static_dd']:.0f}% → {R['vt_dd']:.0f}%**). But when we "
           f"ask *how sure are we?*, the block-bootstrap band on the Sharpe gain is "
           f"**[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]** — it **straddles zero** (the thermostat wins in "
           f"{R['boot_win']:.0f}% of resamples, so ~12% of the time it loses). And the leverage-clean "
           f"significance *t* is **{R['t_alpha']:.2f}** — just under the desk's bar of 2."),
        code(
            "print(f\"bootstrap Sharpe gain {R['boot_point']:+.3f}  95% CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]\")\n"
            "print(f\"  -> straddles zero; vol-target wins {R['boot_win']:.1f}% of resamples\")\n"
            "print(f\"spanning-alpha t = {R['t_alpha']:.2f}  (< 2 -> sub-significant)\")"
        ),
        md("## 3. What *is* rock-solid: the drawdown, and it holds in every crash\n\n"
           f"The one thing that survives every era and every cost is the **shallower drawdown**. In the "
           f"two worst years for the balanced book:\n\n"
           f"- **2008:** {R['crash08_s']:.1f}% → **{R['crash08_v']:.1f}%**\n"
           f"- **2022** (stocks *and* bonds fell together): {R['crash22_s']:.1f}% → **{R['crash22_v']:.1f}%**\n\n"
           "You run this overlay for the *smoother ride*, not for a guaranteed Sharpe pickup."),
        md("## 4. Is the machinery honest? A live synthetic control\n\n"
           "We plant vol-clustering in a seeded toy 60/40, and a flat-vol twin where there's nothing to "
           "forecast. The detector must light up on the first and stay silent on the second. No network."),
        code(
            BOOT +
            "import numpy as np\n"
            "from vt6040 import data, strategy as st\n"
            "planted = st.synthetic_detect(data.synthetic_prices(seed=904, n_days=6000)[0])\n"
            "null    = st.synthetic_detect(data.synthetic_prices(seed=904, n_days=6000, sigma_hi=0.006)[0])\n"
            "print(f\"planted (clustered): Sharpe gain {planted['sharpe_gain']:+.3f}, spanning-alpha t {planted['t_alpha']:+.2f}  (should light up)\")\n"
            "print(f\"null    (flat vol) : Sharpe gain {null['sharpe_gain']:+.3f}, spanning-alpha t {null['t_alpha']:+.2f}  (should be ~0)\")"
        ),
        md("## 5. The verdict\n\n"
           "- **Signal — Weak.** Every point estimate favours the thermostat and the **drawdown "
           "reduction is real and robust** — but the risk-adjusted *improvement* doesn't clear the bar "
           f"(spanning-alpha *t* {R['t_alpha']:.2f}, bootstrap CI straddles zero, and the edge fades from "
           f"+{R['era_e_gain']:.2f} pre-2015 to +{R['era_l_gain']:.2f} after).\n"
           "- **Tradability — Fragile.** You can run it cheaply on the two most liquid ETFs alive, and "
           "the shallower drawdown is bankable — but the thin, decaying Sharpe edge leans on leverage "
           "(avg 1.36×) and a borrow spread eats most of it. A risk overlay you deploy for the ride, "
           "not a certified free lunch."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 893 — Vol-Target 60/40 — the teardown\n\n"
           "The matched-risk race, the **leverage-clean Moreira–Muir spanning alpha** (why *not* a plain "
           "return-difference *t*), the block-bootstrap Sharpe-difference CI, the two-era decay, the "
           "window sweep, the costed timer, and the 30-seed synthetic control."),
        code(R_CELL),
        md("## The race — static vs vol-targeted 60/40 (matched average risk, gross)\n\n"
           "Target = the static blend's own realized vol (10.67%/yr) ⇒ same average risk, only re-timed. "
           "Both books excess-of-cash (minus BIL)."),
        code(
            "print(f\"static  : Sharpe {R['static_sharpe']:.3f}  CAGR {R['static_cagr']:.2f}%  vol {R['static_vol']:.2f}%  maxDD {R['static_dd']:.1f}%\")\n"
            "print(f\"vol-tgt : Sharpe {R['vt_sharpe']:.3f}  CAGR {R['vt_cagr']:.2f}%  vol {R['vt_vol']:.2f}%  maxDD {R['vt_dd']:.1f}%\")\n"
            "print(f\"Sharpe gain {R['sharpe_gain']:+.3f} | spanning alpha {R['alpha_ann']:+.2f}%/yr, HAC t {R['t_alpha']:+.2f} (beta {R['beta']:.3f})\")\n"
            "print(f\"leverage avg {R['avg_lev']:.2f}x, levered {R['frac_lev']}% of days, capped {R['frac_capped']}%, turnover {R['turnover']}x/yr\")"
        ),
        md("### Why the spanning alpha, not a return-difference *t*\n\n"
           f"Because `E[1/σ̂] > 1/E[σ̂]` (Jensen), the thermostat's *average* exposure sits above 1, so a "
           f"plain *t* on the daily return difference (here **{R['diff_t_nw']:+.2f}**) picks up that level "
           f"tilt — it even fires on a flat-vol null. The **leverage-invariant** read is the spanning "
           f"alpha (managed-on-static intercept, HAC *t* = **{R['t_alpha']:+.2f}**) and the Sharpe-*difference* "
           f"bootstrap below."),
        md("## Bootstrap — circular block CI on the excess Sharpe difference (2,000 resamples)"),
        code(
            "print(f\"gain {R['boot_point']:+.3f}  95% CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]  P(vt wins) {R['boot_win']:.1f}%\")\n"
            "print('  -> CI straddles zero: the point estimate is positive, the band is not')"
        ),
        md("## Decay — two eras (split 2015-01-01)"),
        code(
            "print(f\"2007-2014 (n={R['era_e_n']}): gain {R['era_e_gain']:+.3f}  alpha-t {R['era_e_t']:+.2f}  maxDD {R['era_e_dd_s']:.1f}% -> {R['era_e_dd_v']:.1f}%\")\n"
            "print(f\"2015-2026 (n={R['era_l_n']}): gain {R['era_l_gain']:+.3f}  alpha-t {R['era_l_t']:+.2f}  maxDD {R['era_l_dd_s']:.1f}% -> {R['era_l_dd_v']:.1f}%\")\n"
            "print('  Sharpe edge concentrated pre-2015; drawdown cut holds in BOTH eras')"
        ),
        md("## Window sweep + costed timer — no magic point, and does it survive friction?"),
        code(
            "for w,g in [('21d',R['win21']),('42d',R['win42']),('63d',R['win63'])]:\n"
            "    print(f\"window {w}: Sharpe gain {g:+.3f}\")\n"
            "print('costed (one-way bps + 50 bps/yr borrow on the levered fraction):')\n"
            "for c,g in [(0,R['cost0']),(1,R['cost1']),(2,R['cost2']),(5,R['cost5']),(10,R['cost10'])]:\n"
            "    print(f\"  {c:>2} bp: Sharpe gain {g:+.3f}\")\n"
            "print(f\"  drawdown stays ~{R['cost10_dd']:.1f}% even at 10 bp -> the risk-control benefit is robust\")"
        ),
        md("## Synthetic control — the machinery is unbiased (live, offline)\n\n"
           "Portfolio-vol clustering + regime-independent drift is the world where re-timing *should* pay; "
           "flat vol is the null. The detector must fire on one and not the other."),
        code(
            BOOT +
            "import numpy as np\n"
            "from vt6040 import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_prices(seed=893+s, sigma_hi=0.006)[0])['t_alpha'] for s in range(8)])\n"
            "plan_t = np.array([st.synthetic_detect(data.synthetic_prices(seed=893+s)[0])['t_alpha'] for s in range(8)])\n"
            "print(f\"null  (flat vol),  8 seeds: alpha-t mean {null_t.mean():+.2f}, |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "print(f\"planted (clustered), 8 seeds: alpha-t mean {plan_t.mean():+.2f}, t>=2 in {(plan_t>=2).sum()}/8\")\n"
            "print(f\"(frozen 30-seed run: null {R['null_fire']}/{R['n_seeds']} fire, planted {R['plan_fire']}/{R['n_seeds']} fire)\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** Excess Sharpe **{R['static_sharpe']:.3f} → {R['vt_sharpe']:.3f}** and a "
           f"real, robust drawdown cut (**{R['static_dd']:.1f}% → {R['vt_dd']:.1f}%**, both eras, every "
           f"crash) — but the *improvement* is sub-significant: spanning-alpha *t* = **{R['t_alpha']:.2f}** "
           f"(< 2), bootstrap CI **[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]** straddles zero, edge fades "
           f"(+{R['era_e_gain']:.2f} → +{R['era_l_gain']:.2f}) and thins with the window. A single-cycle, "
           f"GFC-anchored, BIL-bounded ~19-year sample.\n"
           f"- **Tradability — Fragile.** Cheap and infinitely scalable on SPY/IEF, and the drawdown "
           f"benefit is bankable — but the thin Sharpe edge is leverage-financed (avg {R['avg_lev']:.2f}×, "
           f"levered {R['frac_lev']}% of days) and a borrow spread eats it from +{R['cost0']:.3f} to "
           f"+{R['cost10']:.3f} by 10 bp. Real but thin, decaying, leverage-dependent -> Fragile."),
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
