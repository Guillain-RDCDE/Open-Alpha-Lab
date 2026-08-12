"""Generate the two narrative notebooks for Study 875 (Idiosyncratic-Vol Change).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; delta-IVOL sort,
# recent-21d minus prior-21d market-model residual vol, long bottom30% / short top30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4104,
    spread_bps=0.87, t_nw=0.86, t_1s=0.85,
    lo_bps=8.43, hi_bps=7.56, welch_t=0.33, gross_sharpe=0.21,
    placebo_obs=0.87, placebo_mean=-0.014, placebo_sd=0.886,
    placebo_p=0.16500, placebo_sigma=0.99, placebo_draws=1000,
    level_bps=-3.85, level_t=-2.85, add_corr=0.216, add_beta=0.158,
    alpha_bps=1.47, alpha_t=1.52,
    era_early_bps=2.86, era_early_t=2.31, era_early_n=1970,
    era_late_bps=-0.98, era_late_t=-0.63, era_late_n=2134,
    timer_1_gross=0.87, timer_1_cost=2.14, timer_1_net=-1.27, timer_1_t=-1.25,
    timer_5_gross=0.87, timer_5_cost=10.14, timer_5_net=-9.27, timer_5_t=-9.12,
    null_mean_t=-0.16, null_sd_t=0.95, null_fire=1,
    planted_t=8.43, planted_welch=4.15,
)


HEADER = f"""# Study 875 — Idiosyncratic-Vol Change 📈

**Does a *rising* idiosyncratic volatility warn of lower future returns?**

The idiosyncratic-vol *level* puzzle (Ang-Hodrick-Xing-Zhang, study 501) sorts on how
**noisy** a name is around the market. This study asks a different question: does the
**change** in that residual noise matter? A **rising** idio-vol — a deteriorating
information environment, rising disagreement — might precede **lower** returns, so a long
**falling-idio-vol** / short **rising-idio-vol** book should earn a positive spread. We
take the self-contained daily version on a liquid US cross-section ({R['start']} →
{R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Strip out each name's market move and you are left with its **idiosyncratic** "
           "return — the wobble that is all its own. Measure how big that wobble is over a "
           "recent month vs the month before: is the name getting **noisier** (idio-vol "
           "rising) or **calmer** (falling)? The story says a rising residual vol flags a "
           "worsening information environment and lower future returns. So buy the names "
           "whose idio-vol is falling, sell the ones where it is rising."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r)\n"
            "print('long falling / short rising delta-IVOL spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  falling-idio-vol book %%+.2f bps vs rising-idio-vol book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world with a real market factor "
           "(`edge>0`) and check the detector recovers it — and that it stays *silent* on "
           "the null (`edge=0`, idio-vol still moves but is unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from ivol_change import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=875, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.002, seed=875, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the change says *nothing* here\n\n"
           f"On this liquid mega-cap tape the long-falling / short-rising delta-IVOL spread "
           f"is **{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"*claimed* sign, but statistically indistinguishable from zero (only "
           f"~{R['placebo_sigma']:.1f}σ into a 1,000-permutation placebo, p = "
           f"{R['placebo_p']:.2f}). Worse, it lives in **one era only**: significant "
           f"*t* = {R['era_early_t']:+.2f} in 2010–2017, then {R['era_late_t']:+.2f} and "
           f"sign-flipped in 2018–2026. And it adds just **{R['alpha_bps']:+.2f} bps/day** "
           f"(*t* = {R['alpha_t']:+.2f}) on top of the idio-vol *level* effect. The seeded "
           "synthetic control recovers a *planted* relation cleanly, so this is a genuine "
           "absence of edge, not a broken sort. **Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 875 — Idiosyncratic-Vol Change — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, the "
           "1,000-permutation placebo, the level-vs-change additivity regression, the "
           "two-era robustness cut, the costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-falling-idio-vol / short-rising-idio-vol spread\n\n"
           "Daily equal-weight bottom-30% minus top-30% delta-IVOL spread. The market "
           "factor is the equal-weight cross-sectional mean return; idio vol is the CAPM "
           "residual vol via `var(r) − cov(r,mkt)²/var(mkt)`."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : falling {R['lo_bps']:+.2f} vs rising {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f}\")\n"
            "print(f\"observed sits {R['placebo_sigma']:+.2f} sigma from the placebo mean\")"
        ),
        md("## Additivity — is the CHANGE anything beyond the idio-vol LEVEL (501)?\n\n"
           "Build the idio-vol *level* sort on the same tape, regress the change spread on "
           "it, and read the residual (Newey-West) *t*."),
        code(
            "print(f\"idio-vol LEVEL spread : {R['level_bps']:+.2f} bps/day (NW t = {R['level_t']:+.2f})\")\n"
            "print(f\"corr(change, level)   : {R['add_corr']:+.3f}   beta = {R['add_beta']:+.3f}\")\n"
            "print(f\"change alpha vs level : {R['alpha_bps']:+.2f} bps/day (NW t = {R['alpha_t']:+.2f})\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)\n\n"
           "The decisive cut: a full-sample *t* carried by one era and reversing in the "
           "other is not a signal."),
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
           "Live: the detector must NOT (reliably) fire on the null and must recover a planted relation."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from ivol_change import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=875+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.002, seed=875, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.002): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The idio-vol *change* — distinct from the idio-vol level "
           f"puzzle (corr with the level spread just {R['add_corr']:+.3f}) — carries **no** "
           f"reliable cross-sectional signal on 50 liquid US mega-caps: the long-falling / "
           f"short-rising spread is **{R['spread_bps']:+.2f} bps/day** (NW *t* = "
           f"**{R['t_nw']:+.2f}**), the *claimed* sign but statistically zero "
           f"(~{R['placebo_sigma']:.1f}σ into the placebo), and — decisively — **not robust "
           f"across eras** (*t* = {R['era_early_t']:+.2f} then {R['era_late_t']:+.2f}, "
           f"sign-flipped). It adds only {R['alpha_bps']:+.2f} bps/day (*t* = "
           f"{R['alpha_t']:+.2f}) on top of the (itself inverted) level effect. The 20-seed "
           f"synthetic control recovers a *planted* relation cleanly (*t* = "
           f"{R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls — the nominal 5%), so "
           f"the flat result is real, not machinery.\n"
           f"- **Tradability — Mirage.** The {R['spread_bps']:+.2f} bps/day gross edge is "
           f"smaller than the {R['timer_1_cost']:.2f} bps/day round-trip friction at 1 bp "
           f"one-way, net **{R['timer_1_net']:+.2f} bps/day** (*t* = {R['timer_1_t']:+.2f}); "
           f"at 5 bps **{R['timer_5_net']:+.2f} bps/day**. Nothing to trade."),
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
