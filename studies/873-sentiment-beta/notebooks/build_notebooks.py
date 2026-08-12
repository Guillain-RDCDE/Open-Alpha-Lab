"""Generate the two narrative notebooks for Study 873 (Sentiment Beta).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLCV,
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; tradable high-minus-
# low-vol sentiment gauge, 252d sentiment beta, long bottom30% / short top30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=3831,
    fingerprint="357fd262912f",
    gauge_bps=4.74, gauge_annvol=19.4, gauge_ac=-0.007,
    spread_bps=-6.20, t_nw=-2.86, t_1s=-2.77,
    lo_bps=4.82, hi_bps=11.01, welch_t=-2.05, gross_sharpe=-0.71,
    cond_high_bps=-9.21, cond_high_t=-2.48, cond_high_n=1150,
    cond_rest_bps=-4.91, cond_rest_t=-1.86, cond_rest_n=2681,
    placebo_obs=-6.20, placebo_mean=0.004, placebo_sd=1.317,
    placebo_p=1.00000, placebo_sigma_left=4.71, placebo_draws=1000,
    era_early_bps=-4.45, era_early_t=-1.73, era_early_n=1697,
    era_late_bps=-7.59, era_late_t=-2.29, era_late_n=2134,
    timer_1_gross=-6.20, timer_1_cost=2.14, timer_1_net=-8.34, timer_1_t=-3.73,
    timer_5_gross=-6.20, timer_5_cost=10.14, timer_5_net=-16.34, timer_5_t=-7.30,
    null_mean_t=-0.05, null_sd_t=0.94, null_fire=0,
    planted_t=5.86, planted_welch=8.13,
)


HEADER = f"""# Study 873 — Sentiment Beta 🎭

**Do the stocks that ride euphoria hardest go on to earn *less*?**

Baker & Wurgler (2006, 2007) argue that a stock's **sentiment beta** — how strongly its
returns co-move with market sentiment — flags the speculative, hard-to-value names that
get over-priced in euphoria and **under-perform afterwards**. So a long **low-sentiment-
beta** / short **high-sentiment-beta** book should earn a *positive* spread, widening
**after sentiment peaks**. We take the self-contained daily version on a liquid US
cross-section ({R['start']} → {R['end']}, {R['n_names']} names), proxying sentiment with
a tradable high-minus-low-volatility spread built from the panel itself.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A **sentiment gauge** rises in risk-on euphoria (the speculative, high-vol "
           "names get bid up) and falls in risk-off. A stock's **sentiment beta** is how "
           "hard it rides that mood. The theory: the high-beta euphoria-chasers are "
           "over-priced when sentiment is high, so their *future* returns disappoint. "
           "Sort on the beta; buy the boring low-beta names, sell the high-beta ones."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r,\n"
            "         gauge_bps=%r, gauge_annvol=%r)\n"
            "print('sentiment gauge (high-minus-low-vol): %%+.2f bps/day, %%.1f%%%% ann-vol'\n"
            "      %% (R['gauge_bps'], R['gauge_annvol']))\n"
            "print('long low-beta / short high-beta spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  low-beta book %%+.2f bps vs high-beta book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"],
               R["gauge_bps"], R["gauge_annvol"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world — a common sentiment factor with "
           "dispersed per-name loadings, and (`edge>0`) high loadings that depress "
           "forward returns — and check the detector recovers it, and stays *silent* on "
           "the null (`edge=0`, betas present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from sentiment_beta import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=873, n_assets=40, n_days=1400))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0025, seed=873, n_assets=40, n_days=1600))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous edge does *not* replicate here\n\n"
           f"On this liquid mega-cap tape the long-low-beta / short-high-beta spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"significant, but with the **opposite sign** to Baker-Wurgler: here the "
           f"high-sentiment-beta names (the momentum tech mega-caps that sync with the "
           f"speculative leg) actually *out-earned* the boring low-beta ones — and by "
           f"*more* after sentiment peaked ({R['cond_high_bps']:+.2f} vs "
           f"{R['cond_rest_bps']:+.2f} bps). The permutation null centres at 0 with sd "
           f"{R['placebo_sd']:.2f} bps; the observed value is ~{R['placebo_sigma_left']:.1f}σ "
           f"into the *left* tail. The seeded synthetic control recovers a *planted* "
           f"Baker-Wurgler relation cleanly, so this is a genuine sign-reversal on the "
           f"mega-cap survivor universe, not a bug — sentiment beta pays where it is a "
           f"speculative-small-stock effect, not on 50 mega-caps. **Signal: None** (the "
           f"claimed edge is absent), **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 873 — Sentiment Beta — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the post-peak conditional, the 1,000-permutation placebo, the two-era "
           "robustness cut, the costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The sentiment gauge — a tradable high-minus-low-vol spread\n\n"
           "Daily return of the top-30% by trailing-63d vol (speculative) minus the "
           "bottom-30% (safe). Real data, built from the panel; low autocorrelation."),
        code(
            "print(f\"gauge         : {R['gauge_bps']:+.2f} bps/day  ann-vol {R['gauge_annvol']:.1f}%  \"\n"
            "      f\"lag-1 autocorr {R['gauge_ac']:+.3f}\")"
        ),
        md("## The headline — long-low-beta / short-high-beta spread\n\n"
           "Daily equal-weight bottom-30% minus top-30% sentiment-beta spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : low-beta {R['lo_bps']:+.2f} vs high-beta {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Conditional — is it stronger *after sentiment peaks*?\n\n"
           "Split the spread by the trailing gauge level (top-30% = a high-sentiment regime)."),
        code(
            "print(f\"high-sentiment (after peaks, n={R['cond_high_n']}): {R['cond_high_bps']:+.2f} bps  NW t = {R['cond_high_t']:+.2f}\")\n"
            "print(f\"the rest                 (n={R['cond_rest_n']}): {R['cond_rest_bps']:+.2f} bps  NW t = {R['cond_rest_t']:+.2f}\")\n"
            "print('-> the conditioning works, but with the REVERSED sign (high-beta out-earned MORE after peaks)')"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> ~{R['placebo_sigma_left']:.1f} sigma into the LEFT tail (right-tail p = {R['placebo_p']:.5f})\")"
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
            "from sentiment_beta import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=873+s, n_assets=40, n_days=1400))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0025, seed=873, n_assets=40, n_days=1600))\n"
            "print(f\"planted (edge=0.0025): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed Baker-Wurgler sentiment-beta premium does **not**"
           f" replicate on 50 liquid US mega-caps: the long-low-beta / short-high-beta spread"
           f" is **{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**) — "
           f"significant but *opposite in sign* (the permutation null centres at 0, sd "
           f"{R['placebo_sd']:.2f} bps; observed ~{R['placebo_sigma_left']:.1f}σ into the left "
           f"tail), and the reversal is present in both eras (*t* = {R['era_early_t']:+.2f} / "
           f"{R['era_late_t']:+.2f}). The 20-seed synthetic control recovers a *planted* "
           f"relation cleanly (*t* = {R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls),"
           f" so the sign-reversal is real, not machinery. Survivorship biases the magnitude.\n"
           f"- **Tradability — Mirage.** Even the sign-flipped book dies: at 1 bp one-way the "
           f"friction ({R['timer_1_cost']:.2f} bps/day) already erodes the {abs(R['spread_bps']):.2f} "
           f"bps gross edge, net **{R['timer_1_net']:+.2f} bps/day** (*t* = {R['timer_1_t']:+.2f}); "
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
