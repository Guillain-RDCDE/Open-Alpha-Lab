"""Generate the two narrative notebooks for Study 803 (Realized-Skewness Reversal).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-21d realized
# skew sort, long bottom30% / short top30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4125,
    spread_bps=-3.19, t_nw=-3.04, t_1s=-2.98,
    lo_bps=6.15, hi_bps=9.34, welch_t=-1.25, gross_sharpe=-0.74,
    placebo_obs=-3.19, placebo_mean=-0.005, placebo_sd=0.873,
    placebo_p=1.00000, placebo_sigma_left=3.65, placebo_draws=1000,
    era_early_bps=-2.37, era_early_t=-2.12, era_early_n=1991,
    era_late_bps=-3.95, era_late_t=-2.27, era_late_n=2134,
    timer_1_gross=-3.19, timer_1_cost=2.14, timer_1_net=-5.33, timer_1_t=-4.98,
    timer_5_gross=-3.19, timer_5_cost=10.14, timer_5_net=-13.33, timer_5_t=-12.46,
    null_mean_t=-0.22, null_sd_t=0.96, null_fire=0,
    planted_t=7.70, planted_welch=7.71,
)


HEADER = f"""# Study 803 — Realized-Skewness Reversal 🎲📉

**Do stocks with a lottery-like recent tape go on to earn *less*?**

Amaya, Christoffersen, Jacobs & Vasquez (2015) find that a stock's **realized
skewness** predicts its cross-section of returns *negatively*: the names whose recent
return distribution is most **right-skewed** under-earn, and a long **low-skew** /
short **high-skew** book earns a positive spread. We take the self-contained daily
version on a liquid US cross-section ({R['start']} → {R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A right-skewed name has a fat *upside* tail — the occasional big up-day. "
           "Skewness-loving investors overpay for that lottery ticket, so the name is "
           "priced too high and its *future* return is lower. Sort on the trailing "
           "third moment; buy the boring left-skewed names, sell the lottery tickets."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r)\n"
            "print('long low-skew / short high-skew spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  low-skew book %%+.2f bps vs high-skew book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`) and check the detector "
           "recovers it — and that it stays *silent* on the null (`edge=0`, skew present "
           "but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from realized_skewness import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=803, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=803, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous edge does *not* replicate here\n\n"
           f"On this liquid mega-cap tape the long-low-skew / short-high-skew spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"significant, but with the **opposite sign** to Amaya et al: here the "
           f"lottery-like high-skew names actually *out-earned* the boring low-skew ones "
           f"(the permutation null centres at 0 with sd {R['placebo_sd']:.2f} bps; the "
           f"observed value is ~{R['placebo_sigma_left']:.1f}σ into the *left* tail). The "
           "seeded synthetic control recovers a *planted* Amaya relation cleanly, so this "
           "is a genuine sign-reversal on the mega-cap survivor universe, not a bug — the "
           "realized-skewness premium is a small-and-illiquid-stock phenomenon that does "
           "not survive on 50 mega-caps. **Signal: None** (the claimed edge is absent), "
           "**Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 803 — Realized-Skewness Reversal — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the two-era robustness cut, the costed timer, "
           "and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-low-skew / short-high-skew spread\n\n"
           "Daily equal-weight bottom-30% minus top-30% realized-skew spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : low-skew {R['lo_bps']:+.2f} vs high-skew {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f}\")"
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
            "from realized_skewness import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=803+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=803, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0016): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed Amaya negative skew→return premium does **not**"
           f" replicate on 50 liquid US mega-caps: the long-low-skew / short-high-skew spread"
           f" is **{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**) — "
           f"significant but *opposite in sign* (the permutation null centres at 0, sd "
           f"{R['placebo_sd']:.2f} bps; observed ~{R['placebo_sigma_left']:.1f}σ into the left "
           f"tail), and it holds in both eras (*t* = {R['era_early_t']:+.2f} / "
           f"{R['era_late_t']:+.2f}). The 20-seed synthetic control recovers a *planted* "
           f"relation cleanly (*t* = {R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls),"
           f" so the sign-reversal is real, not machinery. Survivorship biases the magnitude.\n"
           f"- **Tradability — Mirage.** Even the sign-flipped book dies: at 1 bp one-way the "
           f"friction ({R['timer_1_cost']:.2f} bps/day) already dwarfs the {abs(R['spread_bps']):.2f} "
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
