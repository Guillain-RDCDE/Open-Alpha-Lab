"""Generate the two narrative notebooks for Study 805 (Cokurtosis Premium).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-252d cokurtosis
# with the equal-weight market, long top30% / short bottom30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=3894, n_rows=4147,
    spread_bps=-0.15, t_nw=-0.11, t_1s=-0.11,
    hi_bps=7.39, lo_bps=7.54, welch_t=-0.06, gross_sharpe=-0.03,
    placebo_obs=-0.15, placebo_mean=0.010, placebo_sd=1.131,
    placebo_p=0.54100, placebo_sd_from_null=-0.14, placebo_draws=1000,
    era_early_bps=-1.95, era_early_t=-1.21, era_early_n=1760,
    era_late_bps=1.33, era_late_t=0.61, era_late_n=2134,
    timer_1_gross=-0.15, timer_1_cost=2.14, timer_1_net=-2.29, timer_1_t=-1.61,
    timer_5_gross=-0.15, timer_5_cost=10.14, timer_5_net=-10.29, timer_5_t=-7.26,
    null_mean_t=0.03, null_sd_t=0.90, null_fire=0,
    planted_t=7.20, planted_welch=9.19,
    fingerprint="357fd262912f",
)


HEADER = f"""# Study 805 — Cokurtosis Premium 🪁📐

**Should stocks whose returns amplify the market's fat-tailed moves earn a premium?**

Fang & Lai (1997) extend the CAPM to its *fourth* moment: beyond beta (systematic variance)
and co-skewness (systematic skewness), a name's **cokurtosis with the market** — how strongly
its return co-moves with the *cube* of the market's deviation — is a priced risk. A stock that
spikes exactly when the market has a tail move offers no diversification when you need it most,
so (the theory says) it must pay a **positive** premium: **long high-cokurtosis / short
low-cokurtosis** should earn a positive spread. We take the self-contained daily version on a
liquid US cross-section ({R['start']} → {R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Cokurtosis is a *co-movement* measure, not a stand-alone tail. It asks: on the "
           "days the **market** has an extreme (fat-tailed) move, does *this* name move with "
           "it, hard? A high-cokurtosis name amplifies the market's worst days — it fails to "
           "diversify exactly when diversification matters. Four-moment CAPM says holders of "
           "that exposure should be paid for it. So sort on trailing cokurtosis with the "
           "equal-weight market; buy the high-cokurtosis names, sell the low ones."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r)\n"
            "print('long high-cokurt / short low-cokurt spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  high-cokurt book %%+.2f bps vs low-cokurt book %%+.2f bps'\n"
            "      %% (R['hi_bps'], R['lo_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort even wired up? A live synthetic control\n\n"
           "We plant the premium in a seeded toy world (`knob>0`, high cokurtosis → higher "
           "forward return) and check the detector recovers it — and that it stays *silent* "
           "on the null (`knob=0`, cokurtosis present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from cokurtosis import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(knob=0.0, seed=805, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(knob=0.009, seed=805, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the premium simply isn't there\n\n"
           f"On this liquid mega-cap tape the long-high-cokurt / short-low-cokurt spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — a flat "
           f"**zero**. The high-cokurtosis and low-cokurtosis books earn essentially the same "
           f"return (+{R['hi_bps']:.2f} vs +{R['lo_bps']:.2f} bps/day; Welch *t* = "
           f"{R['welch_t']:+.2f}), the observed spread sits {R['placebo_sd_from_null']:+.2f} sd "
           f"from a 1,000-permutation placebo null, and it even flips sign between eras "
           f"(*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). The seeded synthetic "
           f"control recovers a *planted* premium cleanly (*t* = {R['planted_t']:+.2f}), so this "
           "is a genuine **absence** of the Fang-Lai systematic-kurtosis premium on 50 "
           "mega-caps, not a broken sort. Higher co-moments are famously fragile out of sample; "
           "here the fourth one contributes nothing. **Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 805 — Cokurtosis Premium — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, the "
           "1,000-permutation placebo, the two-era robustness cut, the costed timer, and the "
           "20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-cokurt / short-low-cokurt spread\n\n"
           "Daily equal-weight top-30% minus bottom-30% cokurtosis-with-the-market spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-cokurt {R['hi_bps']:+.2f} vs low-cokurt {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f}\")\n"
            "print(f\"observed sits {R['placebo_sd_from_null']:+.2f} sd from the null (dead centre)\")"
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
           "Live: the detector must NOT fire on the null and must recover a planted premium."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from cokurtosis import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(knob=0.0, seed=805+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (knob=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(knob=0.009, seed=805, n_assets=40, n_days=1500))\n"
            "print(f\"planted (knob=0.009): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The Fang-Lai systematic-kurtosis premium does **not** replicate"
           f" on 50 liquid US mega-caps: the long-high-cokurt / short-low-cokurt spread is "
           f"**{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**) — a flat zero, "
           f"sitting {R['placebo_sd_from_null']:+.2f} sd from a 1,000-permutation placebo null and"
           f" flipping sign across eras (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). "
           f"The 20-seed synthetic control recovers a *planted* premium cleanly (*t* = "
           f"{R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls), so the flat result is a "
           f"real absence, not machinery. Survivorship biases the magnitude only upward.\n"
           f"- **Tradability — Mirage.** There is no gross edge to monetise; the costed book loses "
           f"money ({R['timer_1_net']:+.2f} bps/day at 1 bp one-way, {R['timer_5_net']:+.2f} at "
           f"5 bps). A paycheck from a zero is a Mirage."),
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
