"""Generate the two narrative notebooks for Study 830 (BAB Across Asset Classes).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted
from the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the
fast synthetic positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-
# return closes, nine asset-class ETFs, 2007-04-11 -> 2026-06-30; FP rolling betas to the
# equal-weight multi-asset market, beta-neutral BAB factor).
R = dict(
    start="2007-04-11", end="2026-06-30", n_assets=9, n_rows=4836, n_days=4583,
    fingerprint="c502a478deee",
    bab_bps=0.54, t_nw=0.31, t_1s=0.25, sharpe=0.06,
    alpha_bps=2.86, alpha_t=1.61, realized_beta=-0.83,
    lo_bps=1.60, beta_L=0.56, hi_bps=3.84, beta_H=1.45,
    placebo_obs=0.54, placebo_mean=3.515, placebo_sd=1.528,
    placebo_p=0.9840, placebo_sigma=1.95, placebo_draws=1000,
    era1_bps=3.70, era1_tnw=1.20, era1_alpha_t=1.95, era1_n=2071,
    era2_bps=-2.07, era2_tnw=-1.12, era2_alpha_t=-0.25, era2_n=2512,
    timer_1_gross=0.54, timer_1_cost=0.17, timer_1_net=0.37, timer_1_t=0.17,
    timer_5_gross=0.54, timer_5_cost=0.45, timer_5_net=0.09, timer_5_t=0.04,
    avg_gross=2.62, avg_turnover=0.071,
    null_mean_t=0.12, null_sd_t=0.89, null_fire=1,
    planted_t=3.55, planted_alpha_t=3.92, planted_sharpe=1.06,
)


HEADER = f"""# Study 830 — BAB Across Asset Classes ⚖️🌐

**Does "betting against beta" work when the assets are whole *asset classes*, not stocks?**

Frazzini & Pedersen (2014) show that low-beta stocks earn too much per unit of risk and
high-beta stocks too little — a *flat* security-market line — and package the edge as the
**BAB factor**: long low-beta (levered to unit beta), short high-beta (de-levered to unit
beta), so the book is beta-neutral. The famous claim is that this flat SML is **everywhere**,
including *across asset classes*. We build the multi-asset version on nine liquid asset-class
ETFs ({R['start']} → {R['end']}): equities (SPY/EFA/EEM), bonds & credit (TLT/LQD/HYG), gold
(GLD), commodities (DBC) and REITs (VNQ); beta is measured to their equal-weight portfolio.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. The 'market' is a fixed current-membership ETF basket — named on the
Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "The CAPM says higher beta should pay higher return in a straight line. In "
           "practice the line is too **flat**: leverage-constrained investors crowd into "
           "high-beta assets (bidding their risk-adjusted return down) and shun low-beta "
           "ones. BAB harvests that: buy the calm assets and *lever them up* to market "
           "risk, sell the racy assets and *scale them down*, so market swings cancel and "
           "what is left is the flat-SML alpha. The bet here is that this works not just "
           "among stocks but among **asset classes**."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(bab_bps=%r, t_nw=%r, alpha_bps=%r, alpha_t=%r, realized_beta=%r, sharpe=%r)\n"
            "print('multi-asset BAB factor: %%+.2f bps/day  (Newey-West t = %%+.2f)'\n"
            "      %% (R['bab_bps'], R['t_nw']))\n"
            "print('  CAPM alpha %%+.2f bps/day (HAC t = %%+.2f); realized market beta %%+.2f'\n"
            "      %% (R['alpha_bps'], R['alpha_t'], R['realized_beta']))\n"
            "print('  gross annualized Sharpe: %%.2f' %% R['sharpe'])"
            % (R["bab_bps"], R["t_nw"], R["alpha_bps"], R["alpha_t"],
               R["realized_beta"], R["sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant a flat-SML premium in a seeded toy world (`edge>0`: low-beta assets "
           "carry a positive alpha, high-beta a negative one) and check the detector "
           "recovers it — and stays *silent* on the null (`edge=0`, CAPM holds, betas still "
           "disperse). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from bab_multiasset import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_series(edge=0.0, seed=830, n_days=2500))\n"
            "planted = st.synthetic_detect(data.synthetic_series(edge=0.0006, seed=830, n_days=2500))\n"
            "print('null world   : BAB NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: BAB NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the multi-asset BAB does *not* deliver an alpha here\n\n"
           f"On nine liquid asset classes the beta-neutral BAB factor earns "
           f"**{R['bab_bps']:+.2f} bps/day** with Newey-West *t* = **{R['t_nw']:+.2f}** — "
           f"indistinguishable from zero. Even measured as a CAPM alpha it is only "
           f"**{R['alpha_bps']:+.2f} bps/day (t = {R['alpha_t']:+.2f})**, again short of "
           f"significance, and the book's *realized* market beta is **{R['realized_beta']:+.2f}** "
           f"— far from neutral: the 'low-beta' long leg is dominated by Treasuries and gold, "
           f"so multi-asset BAB is really a disguised **long-duration / short-equity** tilt. "
           f"It worked weakly before 2016 (alpha *t* = {R['era1_alpha_t']:+.2f}) and reversed "
           f"after (*t* = {R['era2_alpha_t']:+.2f}). The seeded synthetic control recovers a "
           f"*planted* flat-SML premium cleanly (*t* = {R['planted_t']:+.2f}), so the machinery "
           f"works — the multi-asset SML just is not flat enough to trade. "
           f"**Signal: None. Tradability: Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 830 — BAB Across Asset Classes — the teardown\n\n"
           "The Frazzini-Pedersen rolling betas to the equal-weight multi-asset market, the "
           "beta-neutral BAB factor, its Newey-West spread *t* and HAC CAPM alpha, the "
           "1,000-permutation placebo, the two-era robustness cut, the costed levered timer, "
           "and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — the beta-neutral BAB factor\n\n"
           "Long low-beta (levered by 1/β_L), short high-beta (de-levered by 1/β_H); "
           "risk-free ≈ 0."),
        code(
            "print(f\"BAB return   : {R['bab_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"CAPM alpha   : {R['alpha_bps']:+.2f} bps/day  HAC t = {R['alpha_t']:+.2f}  \"\n"
            "      f\"(realized market beta {R['realized_beta']:+.2f})\")\n"
            "print(f\"legs         : low-beta {R['lo_bps']:+.2f} bps (beta_L {R['beta_L']:.2f}) vs \"\n"
            "      f\"high-beta {R['hi_bps']:+.2f} bps (beta_H {R['beta_H']:.2f})\")\n"
            "print(f\"gross Sharpe : {R['sharpe']:.2f} (annualized, before cost)\")"
        ),
        md("## Placebo — column-permute the asset returns (1,000 permutations)\n\n"
           "Keep the leverage structure; shuffle *which* asset each leg holds, breaking the "
           "beta→return link. Because the levered book is mechanically ~net-long the average "
           "asset, the permutation null centres **above** zero; the question is whether the "
           "*actual* beta mapping beats a random relabelling."),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> two-sided p = {R['placebo_p']:.4f} \"\n"
            "      f\"({R['placebo_sigma']:.2f} sigma)\")\n"
            "print('  -> the beta sort adds NOTHING beyond the mechanical net-long tilt '\n"
            "      '(observed sits at the low end of the permutation cloud)')"
        ),
        md("## Robustness — two eras (split 2016-07-01)"),
        code(
            "print(f\"2007-2016 (n={R['era1_n']}): {R['era1_bps']:+.2f} bps  NW t = {R['era1_tnw']:+.2f}  alpha t = {R['era1_alpha_t']:+.2f}\")\n"
            "print(f\"2016-2026 (n={R['era2_n']}): {R['era2_bps']:+.2f} bps  NW t = {R['era2_tnw']:+.2f}  alpha t = {R['era2_alpha_t']:+.2f}\")\n"
            "print('  -> weak-only pre-2016, sign-flips after: not a stable edge')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "Realized daily turnover × one-way cost × NAV on the ~2.6×-gross levered book; "
           "short leg pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/day (cost {c:.2f}/day, t={t:+.2f})\")\n"
            "print(f\"  avg gross leverage {R['avg_gross']:.2f}x, turnover {R['avg_turnover']:.3f}/day\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null (CAPM holds) and must recover a "
           "planted flat-SML premium."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from bab_multiasset import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_series(edge=0.0, seed=830+s, n_days=2500))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_series(edge=0.0006, seed=830, n_days=2500))\n"
            "print(f\"planted (edge=0.0006): NW t = {planted['t_nw']:+.2f}, alpha t = {planted['alpha_t']:+.2f}, Sharpe {planted['sharpe']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The multi-asset BAB factor earns **{R['bab_bps']:+.2f} bps/day** "
           f"(NW *t* = **{R['t_nw']:+.2f}**); as a CAPM alpha only **{R['alpha_bps']:+.2f} bps/day** "
           f"(HAC *t* = {R['alpha_t']:+.2f}) — neither clears |t| ≥ 2. The book's realized market "
           f"beta is **{R['realized_beta']:+.2f}** (the 'low-beta' leg is Treasuries + gold, so this "
           f"is a disguised long-duration bet, not a clean SML arbitrage); it worked weakly pre-2016 "
           f"(alpha *t* = {R['era1_alpha_t']:+.2f}) and reversed after ({R['era2_alpha_t']:+.2f}); and "
           f"the 1,000-permutation placebo shows the beta sort adds nothing beyond the mechanical "
           f"net-long tilt (observed {R['placebo_obs']:+.2f} vs cloud mean {R['placebo_mean']:+.2f}, "
           f"two-sided p = {R['placebo_p']:.2f}). The 20-seed synthetic control recovers a *planted* "
           f"flat-SML premium cleanly (*t* = {R['planted_t']:+.2f}, fires {R['null_fire']}/20 nulls ≈ "
           f"the nominal 5%), so the machinery is sound — the cross-asset SML simply is not flat enough.\n"
           f"- **Tradability — Mirage.** The factor is flat gross; net of realized turnover on a "
           f"{R['avg_gross']:.1f}×-gross levered book it is **{R['timer_1_net']:+.2f} bps/day** at 1 bp "
           f"(*t* = {R['timer_1_t']:+.2f}) and **{R['timer_5_net']:+.2f}** at 5 bps — indistinguishable "
           f"from zero at any cost."),
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
