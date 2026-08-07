"""Generate the two narrative notebooks for Study 821 (Turnover Volatility).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC+Volume,
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-63d CV-of-turnover
# sort, long bottom30% / short top30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4084, rows=4147,
    fingerprint="357fd262912f",
    spread_bps=-1.70, t_nw=-1.73, t_1s=-1.67,
    lo_bps=5.95, hi_bps=7.65, welch_t=-0.65, gross_sharpe=-0.41,
    placebo_obs=-1.70, placebo_mean=0.040, placebo_sd=0.931,
    placebo_p=0.97100, placebo_sigma_left=1.87, placebo_draws=1000,
    era_early_bps=-3.00, era_early_t=-2.21, era_early_n=1950,
    era_late_bps=-0.51, era_late_t=-0.36, era_late_n=2134,
    dollar_bps=-1.64, dollar_t=-1.68,
    timer_1_gross=-1.70, timer_1_cost=2.14, timer_1_net=-3.84, timer_1_t=-3.76,
    timer_5_gross=-1.70, timer_5_cost=10.14, timer_5_net=-11.84, timer_5_t=-11.60,
    null_mean_t=0.21, null_sd_t=1.01, null_fire=0,
    planted_t=9.33, planted_welch=9.65,
)


HEADER = f"""# Study 821 — Turnover Volatility 🌀

**Do stocks with the most *erratic* trading go on to earn *less*?**

Chordia, Subrahmanyam & Anshuman (2001) find that beyond the *level* of trading
activity, its **variability** predicts the cross-section of returns *negatively*: the
names whose daily **turnover** is most unpredictable (a high **coefficient of variation**
of turnover) under-earn — a liquidity-risk discount. A long **low**-turnover-vol /
short **high**-turnover-vol book should earn a positive spread. We take the
self-contained daily version on a liquid US cross-section ({R['start']} → {R['end']},
{R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Turnover measures how briskly a stock changes hands. Two names can trade the "
           "*same* average volume, yet one does so like clockwork while the other lurches "
           "between frenzies and droughts. Chordia-Subrahmanyam-Anshuman argue that the "
           "**erratic** one is riskier to hold — you might need to sell exactly when its "
           "liquidity has evaporated — so it is priced at a discount and should *under*-earn. "
           "Sort on the trailing coefficient of variation of turnover; buy the steady "
           "names, sell the erratic ones."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r)\n"
            "print('long low-vol / short high-vol spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  low-turnover-vol book %%+.2f bps vs high-turnover-vol book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: erratic turnover → lower "
           "forward return) and check the detector recovers it — and that it stays *silent* "
           "on the null (`edge=0`, turnover-vol present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from turnover_vol import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=821, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=821, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous edge does *not* replicate here\n\n"
           f"On this liquid mega-cap tape the long-low-vol / short-high-vol spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"statistically **insignificant** (|t| < 2), and if anything faintly the "
           f"*wrong* sign versus the claim (the erratic names slightly *out*-earned). It is "
           f"carried entirely by the pre-2018 era (*t* = {R['era_early_t']:+.2f}) and gone "
           f"thereafter (*t* = {R['era_late_t']:+.2f}); the permutation null sits at zero and "
           f"the observed value is only ~{R['placebo_sigma_left']:.1f}σ from it. The seeded "
           "synthetic control recovers a *planted* CSA relation cleanly, so the machinery is "
           "sound — the turnover-variability premium is a small/illiquid-stock phenomenon "
           "that does not survive on 50 mega-caps. **Signal: None** (the claimed edge is "
           "absent), **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 821 — Turnover Volatility — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the two-era robustness cut, the dollar-volume "
           "variant, the costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-low-vol / short-high-vol spread\n\n"
           "Daily equal-weight bottom-30% minus top-30% turnover-CV spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : low-vol {R['lo_bps']:+.2f} vs high-vol {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.5f} \"\n"
            "      f\"(~{R['placebo_sigma_left']:.2f} sigma into the left tail)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01) + dollar-volume variant"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print(f\"dollar volume (Volume x Close): {R['dollar_bps']:+.2f} bps  NW t = {R['dollar_t']:+.2f}\")"
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
            "from turnover_vol import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=821+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=821, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0016): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed Chordia-Subrahmanyam-Anshuman turnover-variability"
           f" premium does **not** replicate on 50 liquid US mega-caps: the long-low-vol /"
           f" short-high-vol spread is an insignificant **{R['spread_bps']:+.2f} bps/day**"
           f" (NW *t* = **{R['t_nw']:+.2f}**, |t| < 2), faintly *wrong-signed*, carried entirely"
           f" by the pre-2018 era (*t* = {R['era_early_t']:+.2f}) and gone thereafter"
           f" (*t* = {R['era_late_t']:+.2f}); the dollar-volume variant agrees"
           f" ({R['dollar_bps']:+.2f} bps, *t* = {R['dollar_t']:+.2f}) and the placebo shows no"
           f" reliable spread. The 20-seed synthetic control recovers a *planted* relation"
           f" cleanly (*t* = {R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls), so the"
           f" machinery is sound — the effect is simply absent on mega-caps (a small/illiquid"
           f" phenomenon). Survivorship biases the magnitude.\n"
           f"- **Tradability — Mirage.** The specified book loses money gross and net"
           f" ({R['timer_1_net']:+.2f} bps/day at 1 bp, *t* = {R['timer_1_t']:+.2f};"
           f" {R['timer_5_net']:+.2f} at 5 bps); even a sign-flip is eaten by the"
           f" {R['timer_1_cost']:.2f} bps/day round-trip friction at 1 bp."),
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
