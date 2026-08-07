"""Generate the two narrative notebooks for Study 817 (Realized-Volatility Trend).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; vol trend = 21d/63d
# realized-vol ratio - 1, long bottom30% falling / short top30% rising).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4083,
    spread_bps=0.94, t_nw=0.86, t_1s=0.83,
    lo_bps=7.74, hi_bps=6.80, welch_t=0.36, gross_sharpe=0.21,
    placebo_obs=0.94, placebo_mean=-0.021, placebo_sd=0.883,
    placebo_p=0.121, placebo_sigma=1.09, placebo_draws=1000,
    level_bps=-4.74, level_t=-2.61, corr=0.065, beta=0.038,
    alpha_bps=1.12, alpha_t=1.02,
    era_early_bps=1.52, era_early_t=1.15, era_early_n=1949,
    era_late_bps=0.41, era_late_t=0.24, era_late_n=2134,
    timer_1_gross=0.94, timer_1_cost=2.14, timer_1_net=-1.20, timer_1_t=-1.06,
    timer_5_gross=0.94, timer_5_cost=10.14, timer_5_net=-9.20, timer_5_t=-8.14,
    null_mean_t=-0.18, null_sd_t=1.03, null_fire=0,
    planted_t=9.49, planted_welch=9.29,
)


HEADER = f"""# Study 817 — Realized-Volatility Trend 📈📉

**Does *rising* volatility keep de-rating a stock — and *falling* volatility re-rate it?**

Two names can share the same volatility *level* yet be moving in opposite directions:
one's vol is climbing, the other's is cooling. This study sorts on that **trend** —
each name's `(trailing 21d realized vol) / (trailing 63d realized vol) - 1` — long the
**falling-vol** names, short the **rising-vol** ones, and asks the honest question:
is this vol *momentum* anything beyond the low-vol *level* anomaly (study 330)? We take
the daily version on a liquid US cross-section ({R['start']} → {R['end']},
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
           "When a name's *short-window* realized vol pulls **above** its longer-window "
           "average, its risk is being re-priced upward — and, the story goes, the equity "
           "de-rates with it. When vol is **cooling** the opposite: the risk premium "
           "relaxes and the name re-rates. So buy the falling-vol names, sell the "
           "rising-vol ones. Note this is the *change* in vol, deliberately built as a "
           "ratio so it is near-orthogonal to the vol **level** (the low-vol anomaly)."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r,\n"
            "         corr=%r, alpha_bps=%r, alpha_t=%r)\n"
            "print('long falling-vol / short rising-vol spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  falling-vol book %%+.2f bps vs rising-vol book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])\n"
            "print('  corr with low-vol LEVEL sort: %%+.3f  (near-orthogonal)' %% R['corr'])\n"
            "print('  trend alpha net of the level anomaly: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['alpha_bps'], R['alpha_t']))"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"],
               R["corr"], R["alpha_bps"], R["alpha_t"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`) and check the detector "
           "recovers it — and that it stays *silent* on the null (`edge=0`, vol trend "
           "present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from vol_trend import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=817, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0015, seed=817, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the trend says nothing here\n\n"
           f"On this liquid mega-cap tape the long-falling-vol / short-rising-vol spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"*claimed* sign, but statistically indistinguishable from zero (the permutation "
           f"null centres at 0 with sd {R['placebo_sd']:.2f} bps; the observed value is only "
           f"~{R['placebo_sigma']:.1f}σ into the right tail, p = {R['placebo_p']:.2f}). And it "
           f"is **not additive**: near-orthogonal to the low-vol *level* sort "
           f"(corr {R['corr']:+.3f}) yet its alpha net of that level is just "
           f"**{R['alpha_bps']:+.2f} bps/day** (*t* = {R['alpha_t']:+.2f}). The seeded synthetic "
           f"control recovers a *planted* trend relation cleanly, so this is a genuine "
           f"absence of edge, not a bug. **Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 817 — Realized-Volatility Trend — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the level-vs-trend additivity regression, "
           "the two-era robustness cut, the costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-falling-vol / short-rising-vol spread\n\n"
           "Daily equal-weight bottom-30% (falling) minus top-30% (rising) vol-trend spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : falling-vol {R['lo_bps']:+.2f} vs rising-vol {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.3f} \"\n"
            "      f\"({R['placebo_sigma']:+.2f} sigma into the right tail)\")"
        ),
        md("## Additivity — is the vol TREND anything beyond the low-vol LEVEL (330)?\n\n"
           "Build the plain low-vol level sort (long low 63d-vol, short high 63d-vol), then "
           "regress the trend spread on the level spread and read the residual NW *t*."),
        code(
            "print(f\"low-vol LEVEL spread : {R['level_bps']:+.2f} bps/day (NW t = {R['level_t']:+.2f})  \"\n"
            "      f\"# itself inverted on mega-caps\")\n"
            "print(f\"corr(trend, level)   : {R['corr']:+.3f}   beta = {R['beta']:+.3f}  -> near-orthogonal\")\n"
            "print(f\"trend alpha vs level : {R['alpha_bps']:+.2f} bps/day (NW t = {R['alpha_t']:+.2f})  \"\n"
            "      f\"# distinct axis, equally empty\")"
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
            "from vol_trend import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=817+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0015, seed=817, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0015): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The vol *trend* carries no reliable cross-sectional signal on"
           f" 50 liquid US mega-caps: the long-falling-vol / short-rising-vol spread is "
           f"**{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**) — the claimed "
           f"sign but statistically zero, only {R['placebo_sigma']:+.2f}σ into the placebo "
           f"(p = {R['placebo_p']:.2f}), weak in both eras (*t* = {R['era_early_t']:+.2f} / "
           f"{R['era_late_t']:+.2f}). It is **not additive**: near-orthogonal to the level sort "
           f"(corr {R['corr']:+.3f}) yet its alpha net of the level is just "
           f"{R['alpha_bps']:+.2f} bps/day (*t* = {R['alpha_t']:+.2f}). The 20-seed synthetic "
           f"control recovers a *planted* relation cleanly (*t* = {R['planted_t']:+.2f}, fires "
           f"on {R['null_fire']}/20 nulls), so the flat result is real, not machinery.\n"
           f"- **Tradability — Mirage.** The {R['spread_bps']:+.2f} bps/day gross edge is smaller"
           f" than the round-trip friction ({R['timer_1_cost']:.2f} bps/day) at 1 bp one-way, so "
           f"the book is net **{R['timer_1_net']:+.2f} bps/day** (*t* = {R['timer_1_t']:+.2f}); "
           f"at 5 bps **{R['timer_5_net']:+.2f} bps/day**."),
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
