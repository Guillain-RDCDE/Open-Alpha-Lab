"""Generate the two narrative notebooks for Study 898 (Managed-Vol Equity).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY+BIL daily
# total-return, 2007-05-30 -> 2026-06-30; w = min(2.0, 12% / RV_21d), excess-of-cash).
R = dict(
    start="2007-05-30", end="2026-06-30", n_rows=4802, n_days=4780,
    fingerprint="f7ef586be44c",
    sh_strat=0.659, vol_strat=13.4, dd_strat=-31.1, cagr_strat=8.26, wealth_strat=4.51,
    sh_bh=0.549, vol_bh=19.9, dd_bh=-56.5, cagr_bh=9.35, wealth_bh=5.45,
    sharpe_gap=0.110, alpha=2.78, t_alpha=1.67, beta=0.56, appraisal=0.36,
    avg_w=0.96, share_lev=41.5, turnover=9.15,
    exposure_bps=2.407, timing_bps=1.102, diff_bps=-0.822, t_diff=-0.97,
    boot_gap=0.110, boot_lo=-0.128, boot_hi=0.355, boot_pneg=0.195,
    vol_median=12.5, vol_p10=8.8, vol_p90=17.4, vol_band=86.0, bh_vol_p90=27.8,
    era_early_t=0.96, era_early_sh=0.449, era_early_shbh=0.347, era_early_n=2143,
    era_late_t=1.08, era_late_sh=0.827, era_late_shbh=0.759, era_late_n=2637,
    cost1_sh=0.652, cost1_alpha=2.69, cost5_sh=0.613, cost5_alpha=2.16, cost5_dd=-31.4,
    pl_alpha_obs=2.78, pl_alpha_mean=-0.11, pl_alpha_sd=2.03, pl_p_alpha=0.070,
    pl_gap_obs=0.110, pl_gap_mean=-0.055, pl_p_gap=0.050,
    pl_dd_obs=-31.1, pl_dd_mean=-57.4, pl_p_dd=0.000,
    syn_null_t=-0.01, syn_null_fire=7, syn_planted_t=4.98, syn_planted_alpha=9.4,
    syn_planted_fire=97,
    crash=[("GFC 2008-09", -30.6, -56.5), ("2018 Q4", -18.0, -19.8),
           ("COVID 2020", -13.7, -33.9), ("2022 bear", -15.2, -25.0)],
)


HEADER = f"""# Study 898 — Managed-Vol Equity 🎚️

**Turn SPY into a thermostat: aim for a constant ~12% volatility — lean in when the market
is calm, step out when it is stormy. Does that raise the Sharpe, or just tame the ride?**

Moreira & Muir (2017) showed that scaling equity *inversely to recent volatility* raises the
risk-adjusted return. We take the self-contained single-asset version on **SPY vs bills
(BIL)**: hold `w = min(2.0, 12% / RV_21d)` of SPY, the rest in real T-bills, one execution
lag. Sample {R['start']} → {R['end']} ({R['n_rows']:,} daily total-return closes). Every race
is **excess-of-cash on both legs**.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`);
the live cells run the fast synthetic control. Single ~19-year SPY tape — short-history.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Buy-and-hold SPY lets its volatility roam — calm 10% one year, a terrifying 40% "
           "in a crash. A **vol thermostat** rescales daily so the *portfolio* vol stays near "
           "a set point (12%): when SPY gets loud, you hold less of it and more cash; when it "
           "is quiet, you hold a bit more (up to 2×). The pitch is that you shed risk you "
           "were not being paid to bear — a smoother ride, maybe a better Sharpe."),
        code("R = " + repr(R)),
        code(
            "print('MANAGED  vs  BUY & HOLD SPY  (excess of cash, 2007-2026)')\n"
            "print(f\"  Sharpe   {R['sh_strat']:.3f}   vs   {R['sh_bh']:.3f}   (+{R['sharpe_gap']:.3f})\")\n"
            "print(f\"  ann vol  {R['vol_strat']:.1f}%   vs   {R['vol_bh']:.1f}%\")\n"
            "print(f\"  max DD  {R['dd_strat']:.1f}%   vs  {R['dd_bh']:.1f}%   <- the heart-attack cut\")"
        ),
        md("## 2. The heart-attack ledger — where the shield shows up\n\n"
           "The drawdown cut concentrates exactly in the crashes, and it is genuine *timing*: "
           "a 200-seed placebo that shuffles the vol signal (same weight menu, wrong days) "
           "averages a **−57.4%** drawdown — none of 200 shuffles matches the real −31.1% "
           "(**p = 0.000**). Cutting exposure *on the right days* is doing the work."),
        code(
            "for name, s, b in R['crash']:\n"
            "    print(f\"{name:<12}  managed {s:+.1f}%   buy&hold {b:+.1f}%\")\n"
            "print(f\"\\nfull sample   managed {R['dd_strat']:+.1f}%   buy&hold {R['dd_bh']:+.1f}%\")\n"
            "print(f\"placebo (shuffled vol, 200 seeds): mean DD {R['pl_dd_mean']:+.1f}%  ->  p = {R['pl_p_dd']:.3f}\")"
        ),
        md("## 3. Is the sort just lucky? A live synthetic control\n\n"
           "We plant a vol-return *disconnect* in a seeded toy world (`disconnect=2`, mean "
           "falls as variance rises) and check the detector recovers the timing alpha — and "
           "stays *silent* on the null (`disconnect=0`, risk fully priced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from managed_vol import strategy as st\n"
            "# single-seed alpha t is noisy (sd ~1.3), so average a handful of seeds\n"
            "nulls = np.array([st.synthetic_detect(0.0, seed=898+s, n_days=4000)['t_alpha'] for s in range(6)])\n"
            "plant = np.array([st.synthetic_detect(2.0, seed=898+s, n_days=4000)['t_alpha'] for s in range(6)])\n"
            "print(f\"null world   : mean alpha t = {nulls.mean():+.2f}  (should be ~0)\")\n"
            "print(f\"planted world: mean alpha t = {plant.mean():+.2f}  (should light up)\")"
        ),
        md("## 4. The honest verdict — a real shield, an unproven edge\n\n"
           f"On the real SPY tape the thermostat **halves the heart attacks** (max DD "
           f"**{R['dd_strat']:.1f}% vs {R['dd_bh']:.1f}%**) and holds vol near 12% (median "
           f"**{R['vol_median']:.1f}%**) — that half is real and robust. But the celebrated "
           f"**higher-Sharpe** half never certifies: the +{R['sharpe_gap']:.3f} Sharpe gain "
           f"has a bootstrap CI of **[{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]** that straddles "
           f"zero, and the timing alpha is **+{R['alpha']:.2f}%/yr at t = {R['t_alpha']:.2f}** "
           f"(below the *t* ≥ 2 bar). And because the book runs β = {R['beta']} < 1 you *give "
           f"up* ~1.1 pp/yr of excess return for the smoother path. **Signal: Mixed** (real tail "
           f"control, weak Sharpe), **Tradability: Fragile** (cheap to run, but a risk shield, "
           f"not a bankable edge)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 898 — Managed-Vol Equity — the teardown\n\n"
           "The HAC alpha regression, the leverage-timing decomposition, the 3×3 grid, the "
           "paired Sharpe-gap bootstrap, the 200-seed shuffled-signal placebo, the two-era "
           "cut, the cost sweep, and the 30-seed synthetic control. Excess-of-cash on both "
           "legs (SPY − BIL); one documented rebalance lag."),
        code("R = " + repr(R)),
        md("## The headline — managed-vol vs buy-and-hold SPY (excess of cash)"),
        code(
            "print(f\"managed : Sharpe {R['sh_strat']:.3f}  vol {R['vol_strat']:.1f}%  \"\n"
            "      f\"maxDD {R['dd_strat']:.1f}%  excess-CAGR {R['cagr_strat']:.2f}%  x{R['wealth_strat']:.2f}\")\n"
            "print(f\"buy&hold: Sharpe {R['sh_bh']:.3f}  vol {R['vol_bh']:.1f}%  \"\n"
            "      f\"maxDD {R['dd_bh']:.1f}%  excess-CAGR {R['cagr_bh']:.2f}%  x{R['wealth_bh']:.2f}\")\n"
            "print(f\"Sharpe gap {R['sharpe_gap']:+.3f} | HAC alpha {R['alpha']:+.2f}%/yr \"\n"
            "      f\"(t={R['t_alpha']:+.2f}, beta={R['beta']:.2f}, appraisal={R['appraisal']:.2f})\")\n"
            "print(f\"avg weight {R['avg_w']:.2f}, levered {R['share_lev']:.1f}% of days, \"\n"
            "      f\"turnover {R['turnover']:.2f}x NAV/yr, n={R['n_days']}\")"
        ),
        md("## Is it 'just' leverage-timing? The decomposition\n\n"
           "A *constant* scale leaves the Sharpe unchanged, so the whole Sharpe gap **is** the "
           "timing term. Split `mean(managed) = β·mean(B&H) [exposure] + α [timing]`:"),
        code(
            "print(f\"exposure (beta*mean B&H) {R['exposure_bps']:+.3f} bps/day\")\n"
            "print(f\"timing   (alpha)         {R['timing_bps']:+.3f} bps/day\")\n"
            "print(f\"net managed - B&H daily  {R['diff_bps']:+.3f} bps  (HAC t = {R['t_diff']:+.2f})\")\n"
            "print('beta<1: the book GIVES UP raw excess return; the timing alpha buys a lower-vol path.')"
        ),
        md("## Placebo — shuffle the vol signal (200 seeds; same weights, no timing)"),
        code(
            "print(f\"HAC alpha : obs {R['pl_alpha_obs']:+.2f}% vs placebo {R['pl_alpha_mean']:+.2f}% \"\n"
            "      f\"(sd {R['pl_alpha_sd']:.2f}) -> p = {R['pl_p_alpha']:.3f}\")\n"
            "print(f\"Sharpe gap: obs {R['pl_gap_obs']:+.3f} vs placebo {R['pl_gap_mean']:+.3f} -> p = {R['pl_p_gap']:.3f}\")\n"
            "print(f\"max DD    : obs {R['pl_dd_obs']:+.1f}% vs placebo {R['pl_dd_mean']:+.1f}% -> p = {R['pl_p_dd']:.3f}\")\n"
            "print('The tail shield is unambiguous timing (p=0.000); the alpha/Sharpe is only borderline.')"
        ),
        md("## Bootstrap — the Sharpe advantage, paired circular block"),
        code(
            "print(f\"gap {R['boot_gap']:+.3f}  95% CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]  \"\n"
            "      f\"Pr[gap<0] = {R['boot_pneg']:.3f}  -> straddles zero\")"
        ),
        md("## Robustness — two eras (split 2016-01-01)"),
        code(
            "print(f\"2007-2016 (n={R['era_early_n']}): alpha t = {R['era_early_t']:+.2f}  \"\n"
            "      f\"Sharpe {R['era_early_sh']:.3f} (bh {R['era_early_shbh']:.3f})\")\n"
            "print(f\"2016-2026 (n={R['era_late_n']}): alpha t = {R['era_late_t']:+.2f}  \"\n"
            "      f\"Sharpe {R['era_late_sh']:.3f} (bh {R['era_late_shbh']:.3f})\")\n"
            "print('Sign stable in both halves; significance never arrives.')"
        ),
        md("## The timer — costs (one-way bps x |dw| + borrow on the levered leg)"),
        code(
            "print(f\"1 bp        : Sharpe {R['cost1_sh']:.3f}  alpha {R['cost1_alpha']:+.2f}%\")\n"
            "print(f\"5 bp + 1%b  : Sharpe {R['cost5_sh']:.3f}  alpha {R['cost5_alpha']:+.2f}%  maxDD {R['cost5_dd']:.1f}%\")\n"
            "print('Cheap to run; what fails the bar gross also fails it net.')"
        ),
        md("## Synthetic positive control — the machinery is unbiased (live)\n\n"
           "Null (risk priced) must NOT fire; planted leverage-effect must light up."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from managed_vol import strategy as st\n"
            "null_t = np.array([st.synthetic_detect(0.0, seed=898+s, n_days=4000)['t_alpha'] for s in range(8)])\n"
            "plant_t = np.array([st.synthetic_detect(2.0, seed=898+s, n_days=4000)['t_alpha'] for s in range(8)])\n"
            "print(f\"null    (disconnect=0), 8 seeds: alpha t mean {null_t.mean():+.2f} \"\n"
            "      f\"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "print(f\"planted (disconnect=2), 8 seeds: alpha t mean {plant_t.mean():+.2f}, \"\n"
            "      f\"t>=2 in {(plant_t>=2).sum()}/8\")\n"
            "print(f\"(frozen 30-seed run: null mean t {R['syn_null_t']:+.2f} fires {R['syn_null_fire']}%; \"\n"
            "      f\"planted mean t {R['syn_planted_t']:+.2f} fires {R['syn_planted_fire']}%)\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — MIXED (Real on the tail control · Weak on the Sharpe).** The "
           f"drawdown-taming half is real on the real tape: full-sample max DD "
           f"**{R['dd_strat']:.1f}% vs {R['dd_bh']:.1f}%**, robust across all nine grid cells "
           f"and both eras, certified as genuine *timing* by the shuffled-signal placebo "
           f"(**p = {R['pl_p_dd']:.3f}**, 200 seeds); with average weight {R['avg_w']} it "
           f"de-risks in storms, not just holds less. The return half is **not** certified: "
           f"HAC alpha **+{R['alpha']:.2f}%/yr at t = {R['t_alpha']:.2f}** (era t = "
           f"{R['era_early_t']:+.2f}/{R['era_late_t']:+.2f}), and the +{R['sharpe_gap']:.3f} "
           f"Sharpe gain has a bootstrap CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}] "
           f"straddling zero. Single ~19-year SPY tape — short-history, named.\n"
           f"- **Tradability — FRAGILE.** Turnover {R['turnover']:.2f}× NAV/yr costs ~9 bps/yr "
           f"at 1 bp, SPY capacity is unlimited, and every number survives 5 bp + 1% borrow "
           f"(Sharpe {R['cost5_sh']:.3f}). But what survives certification is **risk control, "
           f"not excess return**: β = {R['beta']} < 1 means a ~1.1 pp/yr excess-CAGR give-up "
           f"for the smoother path, and the Sharpe uplift never clears *t* = 2 — a real "
           f"shield, not a bankable edge. FRAGILE, not INVESTABLE."),
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
