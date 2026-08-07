"""Generate the two narrative notebooks for Study 814 (Trailing-Sharpe Anomaly).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing 12-1 Sharpe
# sort, long top30% / short bottom30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=3873,
    spread_bps=1.29, t_nw=0.83, t_1s=0.79,
    hi_bps=8.05, lo_bps=6.75, welch_t=0.46, gross_sharpe=0.20,
    mom_bps=1.59, mom_t=0.99, lowvol_bps=-5.65, lowvol_t=-2.92,
    rho_sharpe_mom=0.953, rho_sharpe_negvol=0.088,
    placebo_obs=1.29, placebo_mean=0.057, placebo_sd=0.986,
    placebo_p=0.096, placebo_sigma=1.25, placebo_draws=1000,
    era_early_bps=0.28, era_early_t=0.15, era_early_n=1739,
    era_late_bps=2.12, era_late_t=0.89, era_late_n=2134,
    timer_1_gross=1.29, timer_1_cost=2.14, timer_1_net=-0.84, timer_1_t=-0.51,
    timer_5_gross=1.29, timer_5_cost=10.14, timer_5_net=-8.84, timer_5_t=-5.40,
    null_mean_t=-0.03, null_sd_t=0.93, null_fire=1,
    planted_t=6.64, planted_welch=6.62,
)


HEADER = f"""# Study 814 — Trailing-Sharpe Anomaly 📐📈

**Does risk-adjusting momentum — ranking on the trailing *Sharpe ratio* — beat plain
momentum, or is it the same trade in a nicer suit?**

Risk-adjusted momentum (Rachev / Biglova et al) replaces Jegadeesh-Titman's raw past-return
ranking with a **reward-to-risk** ranking: each name's **trailing 12-month Sharpe** (mean ÷
std of daily returns, skipping the most recent month). Long the high-Sharpe names, short the
low-Sharpe ones. We take the self-contained daily version on a liquid US cross-section
({R['start']} → {R['end']}, {R['n_names']} names) and ask the honest question head-on:
**does the division earn its keep?**

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea — and the catch\n\n"
           "A Sharpe ratio is *momentum numerator ÷ volatility denominator*. Ranking on it "
           "is supposed to keep the **high-quality** winners and drop the jittery ones. But "
           "notice: if the winners and the high-Sharpe names are mostly the **same** names, "
           "you have not built a new signal — you have re-labelled momentum. So the test is "
           "not 'does the Sharpe book make money' but 'does it **beat plain momentum**?'"),
        code(
            "R = dict(spread_bps=%r, t_nw=%r, mom_bps=%r, mom_t=%r, rho_sharpe_mom=%r,\n"
            "         rho_sharpe_negvol=%r, gross_sharpe=%r)\n"
            "print('long high-Sharpe / short low-Sharpe : %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('plain 12-1 momentum, same sort      : %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['mom_bps'], R['mom_t']))\n"
            "print('rank corr  Sharpe ~ momentum        : %%+.3f' %% R['rho_sharpe_mom'])\n"
            "print('rank corr  Sharpe ~ (-vol)          : %%+.3f' %% R['rho_sharpe_negvol'])"
            % (R["spread_bps"], R["t_nw"], R["mom_bps"], R["mom_t"],
               R["rho_sharpe_mom"], R["rho_sharpe_negvol"], R["gross_sharpe"])
        ),
        md("The Sharpe sort is **0.95 rank-correlated with plain momentum** and carries almost "
           "none of the low-vol tilt (+0.09). It *is* momentum — and it earns a touch **less** "
           "(+1.29 vs +1.59 bps/day). Risk-adjusting bought nothing."),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant a high-Sharpe → high-return effect in a seeded toy world (`edge>0`, via a "
           "persistent volatility tilt) and check the detector recovers it — and stays *silent* "
           "on the null (`edge=0`, Sharpe varies but is unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from trailing_sharpe import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=814, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=814, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — risk-adjusting is not a free lunch\n\n"
           f"On this liquid mega-cap tape the long-high-Sharpe / short-low-Sharpe spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the right "
           f"sign, but **not significant**. And it does not beat what it is built from: at "
           f"**{R['rho_sharpe_mom']:.2f}** rank correlation with plain 12-1 momentum, the Sharpe "
           f"sort simply re-picks the same winners, earning a hair *less* than momentum itself "
           f"(+{R['mom_bps']:.2f} bps, *t* {R['mom_t']:+.2f}) — and neither clears |t| ≥ 2. The "
           f"observed spread sits only ~{R['placebo_sigma']:.1f}σ into a "
           f"{R['placebo_draws']:,}-permutation placebo. The synthetic control recovers a "
           f"*planted* effect cleanly, so this is a real 'nothing here', not a bug. "
           f"**Signal: None** (momentum repackaged), **Tradability: Mirage** (net-negative at "
           f"1 bp)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 814 — Trailing-Sharpe Anomaly — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the Sharpe-vs-momentum-vs-lowvol "
           "head-to-head and rank overlap, the 1,000-permutation placebo, the two-era cut, the "
           "costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-Sharpe / short-low-Sharpe spread\n\n"
           "Daily equal-weight top-30% minus bottom-30% trailing-Sharpe spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-Sharpe {R['hi_bps']:+.2f} vs low-Sharpe {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Does risk-adjusting help? — the head-to-head (the whole point)\n\n"
           "Same universe, same dates, same sort machinery — Sharpe vs plain momentum vs pure "
           "low-vol, plus the average per-day rank overlap of the signals."),
        code(
            "print(f\"trailing Sharpe : {R['spread_bps']:+.2f} bps  NW t = {R['t_nw']:+.2f}\")\n"
            "print(f\"12-1 momentum   : {R['mom_bps']:+.2f} bps  NW t = {R['mom_t']:+.2f}\")\n"
            "print(f\"pure low-vol    : {R['lowvol_bps']:+.2f} bps  NW t = {R['lowvol_t']:+.2f}\")\n"
            "print(f\"rank corr  Sharpe~momentum = {R['rho_sharpe_mom']:+.3f}  \"\n"
            "      f\"Sharpe~(-vol) = {R['rho_sharpe_negvol']:+.3f}\")\n"
            "print('=> 0.95 correlated with momentum, ~0 low-vol content: momentum repackaged.')"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f}  (~{R['placebo_sigma']:.2f} sigma)\")"
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
            "from trailing_sharpe import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=814+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=814, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0016): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The trailing-Sharpe sort earns **{R['spread_bps']:+.2f} "
           f"bps/day** (NW *t* = **{R['t_nw']:+.2f}**) — right sign, not significant. At "
           f"**{R['rho_sharpe_mom']:.2f}** rank correlation with plain 12-1 momentum and only "
           f"+{R['rho_sharpe_negvol']:.2f} with low-vol, it is momentum repackaged and earns a "
           f"touch *less* than the momentum book itself (+{R['mom_bps']:.2f} bps, *t* "
           f"{R['mom_t']:+.2f}); neither clears |t| ≥ 2, it sits ~{R['placebo_sigma']:.1f}σ into "
           f"the placebo, and it is flat in both eras (*t* = {R['era_early_t']:+.2f} / "
           f"{R['era_late_t']:+.2f}). The synthetic control recovers a *planted* effect "
           f"(*t* = {R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls ≈ nominal 5%), so "
           f"the flat real-tape read is genuine. Survivorship biases the magnitude upward.\n"
           f"- **Tradability — Mirage.** The insignificant gross edge goes net-negative at 1 bp "
           f"one-way ({R['timer_1_net']:+.2f} bps/day, *t* = {R['timer_1_t']:+.2f}) as the "
           f"{R['timer_1_cost']:.2f} bps/day friction eats it; at 5 bps **{R['timer_5_net']:+.2f} "
           f"bps/day**."),
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
