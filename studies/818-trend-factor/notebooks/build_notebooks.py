"""Generate the two narrative notebooks for Study 818 (Trend Factor).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic
positive control, so execution is quick and network-free.
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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trend factor = 250d rolling
# Fama-MacBeth slopes on the 7 normalized MA signals, long top30% / short bottom30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=3697,
    spread_bps=1.42, t_nw=0.99, t_1s=0.87,
    lo_bps=7.06, hi_bps=8.48, welch_t=0.49, gross_sharpe=0.23,
    ma200_bps=2.01, ma200_t=1.24, mom_bps=2.24, mom_t=1.40,
    placebo_obs=1.42, placebo_mean=0.019, placebo_sd=0.964,
    placebo_p=0.066, placebo_sigma=1.45, placebo_draws=1000,
    era_early_bps=1.72, era_early_t=0.96, era_early_n=1563,
    era_late_bps=1.20, era_late_t=0.57, era_late_n=2134,
    timer_1_gross=1.42, timer_1_cost=2.14, timer_1_net=-0.72, timer_1_t=-0.44,
    timer_5_gross=1.42, timer_5_cost=10.14, timer_5_net=-8.72, timer_5_t=-5.34,
    null_mean_t=0.07, null_sd_t=0.94, null_fire=1,
    planted_t=10.28, planted_welch=10.64,
)


HEADER = f"""# Study 818 — Trend Factor 🌊

**Does blending *all* the moving-average horizons at once beat any single one — and momentum?**

Han, Zhou & Zhu (2016) build a **trend factor**: for each name form normalized moving-average
signals `A_L = MA_L(price)/price` for `L ∈ {{3,5,10,20,50,100,200}}`, let a rolling cross-
sectional (Fama-MacBeth) regression *weight* those horizons, and dot the averaged past slopes
into today's signals to get a fitted expected return. Sort **long high-trend / short low-trend**.
The paper's headline is that this blend **beats** single-moving-average timing *and* momentum.
We take the self-contained daily version on a liquid US cross-section ({R['start']} → {R['end']},
{R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast synthetic
control. Survivorship: current-membership mega-caps — magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A single moving average looks at *one* time scale. Different traders react over "
           "different horizons — days, weeks, months — so a 200-day rule throws away the "
           "short-horizon information and vice versa. The trend factor lets a cross-sectional "
           "regression *decide the weights* on all seven horizons at once, then buys the names "
           "with the highest fitted expected return and sells the lowest."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r,\n"
            "         ma200_bps=%r, ma200_t=%r, mom_bps=%r, mom_t=%r)\n"
            "print('long high-trend / short low-trend spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  high-trend book %%+.2f bps vs low-trend book %%+.2f bps'\n"
            "      %% (R['hi_bps'], R['lo_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"],
               R["ma200_bps"], R["ma200_t"], R["mom_bps"], R["mom_t"])
        ),
        md("## 2. Is the machinery even wired right? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: persistent trends that both "
           "move the price and predict its next return) and check the fitted trend factor "
           "recovers it — and that it stays *silent* on the null (`edge=0`, prices are random "
           "walks). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from trend_factor import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=818, n_assets=40, n_days=1500), beta_window=120)\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0015, seed=818, n_assets=40, n_days=1500), beta_window=120)\n"
            "print('null world   : trend-factor spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: trend-factor spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The contrast the paper hangs on — and what actually happened\n\n"
           f"The trend factor is *supposed* to beat single-MA timing and momentum. On 50 liquid "
           f"US mega-caps it is the **weakest** of the three:\n\n"
           f"| sort | spread/day | NW *t* |\n|---|--:|--:|\n"
           f"| **trend factor (blend of 7)** | **+{R['spread_bps']:.2f} bps** | **+{R['t_nw']:.2f}** |\n"
           f"| single-MA(200) timing | +{R['ma200_bps']:.2f} bps | +{R['ma200_t']:.2f} |\n"
           f"| 12-1 momentum | +{R['mom_bps']:.2f} bps | +{R['mom_t']:.2f} |\n\n"
           "All three are insignificant, but the fancy blend adds *nothing* over a plain 200-day "
           "rule or vanilla momentum here."),
        md("## 4. The honest verdict — the famous factor does *not* replicate here\n\n"
           f"On this liquid mega-cap tape the long-high-trend / short-low-trend spread is "
           f"**+{R['spread_bps']:.2f} bps/day** with NW *t* = **+{R['t_nw']:.2f}** — the right "
           f"sign but statistically indistinguishable from zero (only ~{R['placebo_sigma']:.2f} sd "
           f"into a {R['placebo_draws']:,}-permutation placebo, p = {R['placebo_p']:.3f}), weak "
           f"in both eras (*t* = +{R['era_early_t']:.2f} / +{R['era_late_t']:.2f}). The seeded "
           f"synthetic control recovers a *planted* trend relation cleanly (*t* = "
           f"+{R['planted_t']:.2f}), so this is a genuine null on the mega-cap survivor slice, not "
           f"a bug — the trend premium is a broad-cross-section / small-cap phenomenon. And the "
           f"book dies under the lightest costs: at 1 bp one-way, net **{R['timer_1_net']:+.2f} "
           f"bps/day**. **Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 818 — Trend Factor — the teardown\n\n"
           "The headline spread's Newey-West *t*, the pooled Welch book test, the single-MA and "
           "momentum contrast, the 1,000-permutation placebo, the two-era robustness cut, the "
           "costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-trend / short-low-trend spread\n\n"
           "Daily equal-weight top-30% minus bottom-30% trend-factor spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-trend {R['hi_bps']:+.2f} vs low-trend {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:+.2f} (before cost)\")"
        ),
        md("## Contrast — the two sorts the trend factor is *claimed to beat*\n\n"
           "The paper's central claim is that the blend dominates single-MA timing and momentum. "
           "Here it is the weakest of the three."),
        code(
            "print(f\"trend factor (blend) : {R['spread_bps']:+.2f} bps  NW t = {R['t_nw']:+.2f}\")\n"
            "print(f\"single-MA(200) timing: {R['ma200_bps']:+.2f} bps  NW t = {R['ma200_t']:+.2f}\")\n"
            "print(f\"12-1 momentum        : {R['mom_bps']:+.2f} bps  NW t = {R['mom_t']:+.2f}\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.3f} ({R['placebo_sigma']:+.2f} sd from mean)\")"
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
           "Live: the fitted trend factor must NOT fire on the null and must recover a planted relation."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from trend_factor import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=818+s, n_assets=40, n_days=1500), beta_window=120)['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0015, seed=818, n_assets=40, n_days=1500), beta_window=120)\n"
            "print(f\"planted (edge=0.0015): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed Han-Zhou-Zhu trend factor does **not** replicate on "
           f"50 liquid US mega-caps: the long-high-trend / short-low-trend spread is "
           f"**+{R['spread_bps']:.2f} bps/day** (NW *t* = **+{R['t_nw']:.2f}**) — the right sign "
           f"but indistinguishable from zero, only ~{R['placebo_sigma']:.2f} sd into the "
           f"permutation null (p = {R['placebo_p']:.3f}), weak in both eras "
           f"(*t* = +{R['era_early_t']:.2f} / +{R['era_late_t']:.2f}), and — fatally for the claim "
           f"— **weaker than the single-MA(200) (*t* = +{R['ma200_t']:.2f}) and momentum "
           f"(*t* = +{R['mom_t']:.2f}) sorts it is supposed to beat**. The 20-seed synthetic "
           f"control recovers a *planted* relation cleanly (*t* = +{R['planted_t']:.2f}, fires on "
           f"{R['null_fire']}/20 nulls), so this is a true null, not machinery. Survivorship "
           f"biases the magnitude upward.\n"
           f"- **Tradability — Mirage.** The +{R['spread_bps']:.2f} bps/day gross edge is smaller "
           f"than the {R['timer_1_cost']:.2f} bps/day round-trip friction at a mere 1 bp one-way, "
           f"so the book is net **{R['timer_1_net']:+.2f} bps/day** (*t* = {R['timer_1_t']:+.2f}); "
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
