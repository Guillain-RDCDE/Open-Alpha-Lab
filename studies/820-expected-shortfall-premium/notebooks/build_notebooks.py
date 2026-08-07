"""Generate the two narrative notebooks for Study 820 (Expected-Shortfall Premium).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-252d
# Expected Shortfall (5%) sort, long top30% high-ES / short bottom30% low-ES).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=3894,
    spread_bps=5.41, t_nw=2.80, t_1s=2.71,
    hi_bps=10.43, lo_bps=5.03, welch_t=1.84, gross_sharpe=0.69,
    placebo_obs=5.41, placebo_mean=-0.072, placebo_sd=1.147,
    placebo_p=0.00000, placebo_sigma_right=4.78, placebo_draws=1000,
    era_early_bps=3.86, era_early_t=1.67, era_early_n=1760,
    era_late_bps=6.68, era_late_t=2.26, era_late_n=2134,
    timer_1_gross=5.41, timer_1_cost=2.14, timer_1_net=3.27, timer_1_t=1.64, timer_1_sharpe=0.42,
    timer_5_gross=5.41, timer_5_cost=10.14, timer_5_net=-4.73, timer_5_t=-2.37,
    null_mean_t=0.03, null_sd_t=0.97, null_fire=2,
    planted_t=3.54, planted_welch=3.64,
)


HEADER = f"""# Study 820 — Expected-Shortfall Premium 🎯📉

**Do stocks with a fat left tail go on to earn *more*?**

A downside **tail-risk premium**: a stock whose recent daily returns carry a large
**Expected Shortfall** (CVaR at 5% — the mean of its *worst 5%* of days) is exposed to
deeper crashes, so if that tail risk is *priced* it should be compensated with a higher
future return. Sort long **high-ES** / short **low-ES**. We take the self-contained
daily version on a liquid US cross-section ({R['start']} → {R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — the bias flatters this
sort, so magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "**Expected Shortfall** answers *'on my worst days, how bad is bad?'* — it is the "
           "average of the deepest 5% of daily losses. A fat left tail is real crash exposure; "
           "a rational market should pay you to hold it. So rank the cross-section on trailing "
           "one-year ES; buy the fat-tail names, sell the calm ones, and see if the risk is "
           "compensated."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r)\n"
            "print('long high-ES / short low-ES spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  high-ES book %%+.2f bps vs low-ES book %%+.2f bps'\n"
            "      %% (R['hi_bps'], R['lo_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`, fat left tail → higher mean) "
           "and check the detector recovers it — and that it stays *silent* on the null "
           "(`edge=0`, tail fatness present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from expected_shortfall import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=820, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0024, seed=820, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — real in sign, but a *volatility* sort in disguise\n\n"
           f"On this liquid mega-cap tape the long-high-ES / short-low-ES spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the claimed "
           f"sign *holds*, and the observed value sits ~{R['placebo_sigma_right']:.1f}σ into the "
           f"**right** tail of a 1,000-permutation placebo. But read the fine print: Expected "
           f"Shortfall is ~collinear with volatility, so this is essentially a "
           f"**long-high-vol / short-low-vol** sort — the *opposite* of the low-vol anomaly — and "
           f"on a **survivor** universe it just says the high-vol tech mega-caps (NVDA, TSLA, AMD) "
           f"won 2010–2026. It is era-dependent (*t* = {R['era_early_t']:+.2f} pre-2018 vs "
           f"{R['era_late_t']:+.2f} after, significant in only one half). **Signal: Weak**, **Tradability: "
           f"Fragile** — at 1 bp it nets only {R['timer_1_net']:+.2f} bps/day (*t* = "
           f"{R['timer_1_t']:+.2f}) and dies at 5 bps."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 820 — Expected-Shortfall Premium — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the two-era robustness cut, the costed timer, "
           "and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-ES / short-low-ES spread\n\n"
           "Daily equal-weight top-30% (high ES) minus bottom-30% (low ES) Expected-Shortfall spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-ES {R['hi_bps']:+.2f} vs low-ES {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f}  \"\n"
            "      f\"(~{R['placebo_sigma_right']:+.2f} sigma, right tail)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}  (not sig alone)\")\n"
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
            "from expected_shortfall import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=820+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0024, seed=820, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0024): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak (a flattered near-miss).** The claimed downside tail-risk premium "
           f"**replicates in sign** on 50 liquid US mega-caps: the long-high-ES / short-low-ES "
           f"spread is **{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**), "
           f"~{R['placebo_sigma_right']:.1f}σ into the right tail of a {R['placebo_draws']:,}-permutation "
           f"placebo, with a clean 20-seed synthetic control (planted *t* = {R['planted_t']:+.2f}, "
           f"fires on {R['null_fire']}/20 nulls ≈ nominal). But it is **era-dependent** "
           f"(*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}), the pooled book test is "
           f"sub-threshold (Welch *t* = {R['welch_t']:+.2f}), and — because ES is ~collinear with "
           f"volatility on a survivor universe — it is most honestly the surviving high-vol "
           f"mega-caps winning, the *inverse* of the low-vol anomaly, not a clean priced tail premium.\n"
           f"- **Tradability — Fragile.** At 1 bp one-way the book nets {R['timer_1_net']:+.2f} bps/day "
           f"(~+8%/yr, Sharpe {R['timer_1_sharpe']:.2f}) but net *t* = {R['timer_1_t']:+.2f} (< 2); at "
           f"5 bps the friction ({R['timer_5_cost']:.2f} bps/day) swamps it, net **{R['timer_5_net']:+.2f} "
           f"bps/day** (*t* = {R['timer_5_t']:+.2f}). Real gross edge, too thin to trade."),
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
