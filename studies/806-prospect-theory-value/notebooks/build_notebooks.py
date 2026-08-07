"""Generate the two narrative notebooks for Study 806 (Prospect-Theory Value).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-1260d TK-value
# sort, monthly rebalance, long bottom30% low-TK / short top30% high-TK, 137 months).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50,
    sort_start="2015-01", sort_end="2026-05", n_months=137,
    spread_bps=138.70, t_nw=3.15, t_1s=2.87,
    lo_bps=232.81, hi_bps=94.11, welch_t=2.05, gross_sharpe=0.85,
    placebo_obs=138.70, placebo_mean=0.166, placebo_sd=33.869,
    placebo_p=0.00000, placebo_sigma=4.09, placebo_draws=1000,
    era_early_bps=100.35, era_early_t=2.00, era_early_n=60,
    era_late_bps=168.58, era_late_t=2.48, era_late_n=77,
    timer_1_gross=138.70, timer_1_cost=6.17, timer_1_net=132.53, timer_1_t=2.74,
    timer_1_sharpe=0.81, timer_1_ann=15.9,
    timer_5_gross=138.70, timer_5_cost=14.17, timer_5_net=124.53, timer_5_t=2.57,
    timer_5_sharpe=0.76, timer_5_ann=14.9,
    null_mean_t=-0.11, null_sd_t=1.01, null_fire=0,
    planted_t=2.82, planted_welch=2.40,
)


HEADER = f"""# Study 806 — Prospect-Theory Value 🧠🎰

**Do stocks that look like an attractive *gamble* go on to earn *less*?**

Barberis, Mukherjee & Wang (2016) find that the **cumulative-prospect-theory (TK) value** of a
stock's recent return distribution predicts its cross-section of returns *negatively*: names
whose recent tape looks like a good gamble under Tversky-Kahneman — a right-skewed, lottery-like
distribution — carry a **high** TK value, are over-priced, and go on to under-earn. A long
**low-TK** / short **high-TK** book should earn a positive spread. We take the self-contained
daily version on a liquid US cross-section ({R['start']} → {R['end']}, {R['n_names']} names,
trailing ≈5y TK value, monthly rebalance).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A prospect-theory investor evaluating a single stock as a standalone gamble uses a "
           "value function that is concave over gains, steeper over losses (loss aversion 2.25), "
           "and **probability weights** that overweight the tails. A right-skewed, lottery-like "
           "tape puts mass in the overweighted *upside* tail, so its **TK value is high** — the "
           "investor over-pays, and the future return is lower. Sort on the TK value; buy the "
           "boring low-TK names, sell the lottery tickets."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r)\n"
            "print('long low-TK / short high-TK spread: %%+.2f bps/month (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  low-TK book %%+.2f bps vs high-TK book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost, ann.): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: a lottery-like tape both scores "
           "high TK *and* earns less) and check the detector recovers it — and that it stays "
           "*silent* on the null (`edge=0`, TK varies but is unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from prospect_theory import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=806, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0020, seed=806, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous edge *replicates* here\n\n"
           f"On this liquid mega-cap tape the long-low-TK / short-high-TK spread is "
           f"**{R['spread_bps']:+.2f} bps/month** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"significant and with the **sign prospect theory predicts**: the boring low-TK names "
           f"out-earned the lottery-like high-TK names over 2015–2026 (the permutation null "
           f"centres at 0 with sd {R['placebo_sd']:.1f} bps; the observed value is "
           f"~{R['placebo_sigma']:.1f}σ into the *right* tail). It holds in both eras "
           f"(*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}), and the seeded synthetic "
           f"control recovers a planted relation cleanly — so this is a genuine replication. The "
           f"spread even clears conservative costs (net **{R['timer_5_net']:+.2f} bps/month** at "
           f"5 bps), but it leans on a survivorship-inflated short leg (the blown-up lottery names "
           f"are absent) and a few hard-to-borrow lottery mega-caps. **Signal: Real**, "
           f"**Tradability: Fragile.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 806 — Prospect-Theory Value — the teardown\n\n"
           "The TK value math, the per-leg splits, the Newey-West spread *t*, the pooled Welch "
           "book test, the 1,000-permutation placebo, the two-era robustness cut, the costed "
           "timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The TK value function — a quick live sanity check\n\n"
           "A right-skewed, lottery-like distribution must score a **higher** TK value than its "
           "left-skewed mirror; the decision weights are non-negative and subadditive."),
        code(
            "import os, sys, numpy as np\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from prospect_theory import strategy as st\n"
            "rng = np.random.default_rng(2)\n"
            "lottery = np.concatenate([rng.normal(-0.004,0.008,480), rng.normal(0.12,0.02,20)])\n"
            "crash = -lottery\n"
            "print(f'TK(lottery, right-skew) = {st.tk_value(lottery):+.4f}  (HIGH -> over-priced)')\n"
            "print(f'TK(crash,   left-skew)  = {st.tk_value(crash):+.4f}  (LOW)')\n"
            "assert st.tk_value(lottery) > st.tk_value(crash)"
        ),
        md("## The headline — long-low-TK / short-high-TK spread\n\n"
           "Monthly equal-weight bottom-30% (low TK) minus top-30% (high TK) spread, 137 months."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/month  NW(6) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : low-TK {R['lo_bps']:+.2f} vs high-TK {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (ann., before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f} \"\n"
            "      f\"({R['placebo_sigma']:+.2f} sigma into the right tail)\")"
        ),
        md("## Robustness — two eras (split 2020-01-01)"),
        code(
            "print(f\"2015-2019 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2020-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per monthly rebalance; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t,sh,an in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t'],R['timer_1_sharpe'],R['timer_1_ann']),\n"
            "                          ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'],R['timer_5_sharpe'],R['timer_5_ann'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/mo (cost {c:.2f}/reb, t={t:+.2f}, Sharpe {sh:.2f}, ~{an:+.1f}%/yr)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted relation."),
        code(
            "import numpy as np\n"
            "from prospect_theory import data\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=806+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0020, seed=806, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0020): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The Barberis-Mukherjee-Wang prospect-theory-value premium"
           f" **replicates** on 50 liquid US mega-caps with the **predicted sign**: the"
           f" long-low-TK / short-high-TK spread is **{R['spread_bps']:+.2f} bps/month** (NW *t* ="
           f" **{R['t_nw']:+.2f}**), holds in both eras (*t* = {R['era_early_t']:+.2f} /"
           f" {R['era_late_t']:+.2f}), sits ~{R['placebo_sigma']:+.1f}σ into the right tail of a"
           f" 1,000-permutation placebo, and the 20-seed synthetic control recovers a *planted*"
           f" relation cleanly (*t* = {R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls)."
           f" Survivorship biases the magnitude (upper bound).\n"
           f"- **Tradability — Fragile.** The net edge *survives* conservative costs (net"
           f" **{R['timer_5_net']:+.2f} bps/month** at 5 bps one-way, *t* = {R['timer_5_t']:+.2f},"
           f" ~{R['timer_5_ann']:+.1f}%/yr) — not a Mirage — but the magnitude is a survivorship"
           f" upper bound (the short leg's blown-up lottery names are absent) and the 50-name"
           f" universe concentrates the short into a few hard-to-borrow lottery mega-caps whose"
           f" realistic borrow/squeeze exceeds the 50 bps/yr charged. Real signal, fragile"
           f" paycheck."),
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
