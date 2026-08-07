"""Generate the two narrative notebooks for Study 822 (Omega-Ratio Sort).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC,
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing 12-1 Omega(0)
# sort, long top30% / short bottom30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=3873,
    spread_bps=1.19, t_nw=0.76, t_1s=0.72,
    hi_bps=7.93, lo_bps=6.74, welch_t=0.42, gross_sharpe=0.18,
    omega_bps=1.19, omega_t=0.76, sharpe_bps=1.29, sharpe_t=0.83,
    lowvol_bps=-5.65, lowvol_t=-2.92,
    rho_omega_sharpe=0.996, rho_omega_negvol=0.075,
    placebo_obs=1.19, placebo_mean=0.087, placebo_sd=0.980,
    placebo_p=0.120, placebo_draws=1000,
    era_early_bps=0.29, era_early_t=0.16, era_early_n=1739,
    era_late_bps=1.92, era_late_t=0.80, era_late_n=2134,
    timer_1_gross=1.19, timer_1_cost=2.14, timer_1_net=-0.95, timer_1_t=-0.58,
    timer_5_gross=1.19, timer_5_cost=10.14, timer_5_net=-8.95, timer_5_t=-5.44,
    null_mean_t=0.05, null_sd_t=1.00, null_fire=1,
    planted_t=9.67, planted_welch=9.49,
)


HEADER = f"""# Study 822 — Omega-Ratio Sort ⚖️

**Does a full gain/loss ratio beat plain trailing Sharpe?**

Keating & Shadwick (2002) sell the **Omega ratio** — `Ω(0) = E[max(r,0)] / E[max(−r,0)]`,
the ratio of a name's average gain to its average loss — as a "universal" performance
measure that reads the *whole* return distribution (every moment, both tails), unlike the
Sharpe ratio which stops at mean and variance. The pitch: sort a cross-section on trailing
Omega (long high-Omega / short low-Omega) and this richer, distribution-aware sort should
beat a plain trailing-Sharpe sort ([study 814](../../814-trailing-sharpe-anomaly/)). We put
that head-to-head on a liquid US cross-section ({R['start']} → {R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Sharpe divides a name's average return by its volatility — two numbers, and it's "
           "blind to *shape*. The **Omega ratio** at 0 instead adds up all the up-day returns "
           "and divides by all the down-day losses: `Ω(0) = avg gain / avg loss`. It 'sees' "
           "skewness and fat tails. The claim is that this fuller picture should pick better "
           "stocks than Sharpe. We test it — and check whether Omega is really any different "
           "from Sharpe once you point both at real returns."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(omega_bps=%r, omega_t=%r, sharpe_bps=%r, sharpe_t=%r,\n"
            "         rho_omega_sharpe=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r)\n"
            "print('long high-Omega / short low-Omega spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['omega_bps'], R['omega_t']))\n"
            "print('  vs the plain trailing-Sharpe sort       : %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['sharpe_bps'], R['sharpe_t']))\n"
            "print('  Omega ~ Sharpe per-day rank correlation  : %%+.3f  (near-identical sorts)'\n"
            "      %% R['rho_omega_sharpe'])\n"
            "print('  gross spread Sharpe (before cost)        : %%.2f' %% R['gross_sharpe'])"
            % (R["omega_bps"], R["omega_t"], R["sharpe_bps"], R["sharpe_t"],
               R["rho_omega_sharpe"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just noise? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: low-vol / high-Omega names "
           "really do out-earn) and check the detector recovers it — and that it stays "
           "*silent* on the null (`edge=0`, Omega varies but predicts nothing). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from omega_ratio import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=822, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=822, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — Omega does *not* beat Sharpe here\n\n"
           f"On this liquid mega-cap tape the long-high-Omega / short-low-Omega spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the right "
           f"sign, but nowhere near the |*t*| ≥ 2 significance bar. And the head-to-head is "
           f"brutal: the Omega sort is **{R['rho_omega_sharpe']:.3f} rank-correlated with the "
           f"plain Sharpe sort** and earns *less* than it ({R['sharpe_bps']:+.2f} bps). The "
           f"'whole distribution' Omega is sold on buys nothing over mean/vol here — on daily "
           f"equity returns `Ω(0)` is just a re-labelling of Sharpe, and it inherits Sharpe's "
           f"insignificance. The seeded synthetic control fires cleanly on a *planted* effect, "
           f"so this is a genuine null, not a broken sort. **Signal: None**, "
           f"**Tradability: Mirage** (the thin edge dies at any realistic cost)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 822 — Omega-Ratio Sort — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the Omega-vs-Sharpe-vs-low-vol "
           "head-to-head with rank overlaps, the 1,000-permutation placebo, the two-era "
           "robustness cut, the costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-Omega / short-low-Omega spread\n\n"
           "Daily equal-weight top-30% minus bottom-30% trailing-Omega(0) spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-Omega {R['hi_bps']:+.2f} vs low-Omega {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Does the full gain/loss ratio beat Sharpe? — the head-to-head\n\n"
           "Same universe, same dates, same sort machinery. If Omega's extra moments matter, "
           "it should out-earn Sharpe and pick different names."),
        code(
            "print(f\"Omega  (long high / short low): {R['omega_bps']:+.2f} bps  NW t = {R['omega_t']:+.2f}\")\n"
            "print(f\"Sharpe (long high / short low): {R['sharpe_bps']:+.2f} bps  NW t = {R['sharpe_t']:+.2f}\")\n"
            "print(f\"low-vol(long low / short high): {R['lowvol_bps']:+.2f} bps  NW t = {R['lowvol_t']:+.2f}\")\n"
            "print(f\"rank corr  Omega~Sharpe = {R['rho_omega_sharpe']:+.3f}   Omega~(-vol) = {R['rho_omega_negvol']:+.3f}\")\n"
            "print('=> Omega is ~identical to Sharpe (rho +0.996) and does NOT beat it; the low-vol confound is NOT the driver.')"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.3f}\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per day on the long-short book; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/day (cost {c:.2f}/day, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted relation."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from omega_ratio import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=822+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=822, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0016): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The Keating-Shadwick Omega advantage does **not** materialise "
           f"on 50 liquid US mega-caps: the long-high-Omega / short-low-Omega spread is "
           f"**{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**) — right sign, "
           f"insignificant, flat in both eras (*t* = {R['era_early_t']:+.2f} / "
           f"{R['era_late_t']:+.2f}), placebo p = {R['placebo_p']:.2f}. It is "
           f"**{R['rho_omega_sharpe']:.3f} rank-identical to a plain Sharpe sort** and does not "
           f"beat it ({R['sharpe_bps']:+.2f} bps) — the extra moments add nothing. The 20-seed "
           f"synthetic control recovers a *planted* effect cleanly (*t* = {R['planted_t']:+.2f}, "
           f"fires on {R['null_fire']}/20 nulls), so this is a true null.\n"
           f"- **Tradability — Mirage.** The book is net-negative at every realistic cost: at "
           f"1 bp one-way the friction ({R['timer_1_cost']:.2f} bps/day) already exceeds the "
           f"{R['spread_bps']:.2f} bps gross edge, net **{R['timer_1_net']:+.2f} bps/day**; at "
           f"5 bps **{R['timer_5_net']:+.2f} bps/day**."),
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
