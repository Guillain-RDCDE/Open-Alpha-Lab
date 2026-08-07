"""Generate the two narrative notebooks for Study 811 (Zero-Return Illiquidity).

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
# zero-return-proportion sort, long top30% illiquid / short bottom30% liquid).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=3894, fingerprint="357fd262912f",
    zp_min_pct=0.00, zp_med_pct=0.00, zp_max_pct=2.38, long_zp_pct=1.38,
    spread_bps=-1.37, t_nw=-1.29, t_1s=-1.29,
    hi_bps=6.55, lo_bps=7.93, welch_t=-0.52, gross_sharpe=-0.33,
    placebo_obs=-1.37, placebo_mean=0.123, placebo_sd=1.108,
    placebo_p=0.907, placebo_sigma_left=1.35, placebo_draws=1000,
    era_early_bps=-2.23, era_early_t=-1.69, era_early_n=1760,
    era_late_bps=-0.66, era_late_t=-0.41, era_late_n=2134,
    timer_1_gross=-1.37, timer_1_cost=2.14, timer_1_net=-3.51, timer_1_t=-3.30,
    timer_5_gross=-1.37, timer_5_cost=10.14, timer_5_net=-11.51, timer_5_t=-10.83,
    null_mean_t=-0.67, null_sd_t=0.88, null_fire=0,
    planted_t=11.23, planted_welch=10.98,
)


HEADER = f"""# Study 811 — Zero-Return Illiquidity 🕳️

**Do stocks that print *exactly-zero* daily returns earn an illiquidity premium?**

Lesmond, Ogden & Trzcinka (1999) argue that when the round-trip cost of trading
exceeds the day's information, the informed trader stays home and the price **doesn't
move** — so the *frequency of zero-return days* is a cheap, price-only proxy for a
name's transaction cost. Illiquid names should be compensated (Amihud & Mendelson
1986), so a long **high-zero** / short **low-zero** book should earn a positive spread.
We take the self-contained daily version on a liquid US cross-section ({R['start']} →
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
           "A stock nobody trades sits still: with no trade, the close repeats and the "
           "day's return is **exactly zero**. Count how often that happens over the past "
           "year and you have a free illiquidity meter — the harder a name is to trade, "
           "the more zero-days it prints. Illiquidity should be *paid for* (Amihud & "
           "Mendelson), so buy the often-zero names, sell the never-zero ones."),
        md("## 2. The catch, before we even look at returns\n\n"
           "This trick was built for tick-priced, thinly-traded small-caps. **Mega-caps "
           "almost never sit still.** On our 50-name universe the *median* stock prints "
           f"a zero return on **{R['zp_med_pct']:.2f}%** of trailing-year days, and the "
           f"single most-frequent name only reaches **{R['zp_max_pct']:.2f}%**. Half the "
           "list is pinned at exactly 0.00% — the sort has almost nothing to bite on. "
           "That alone tells us to expect **None** here."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r,\n"
            "         zp_med_pct=%r, zp_max_pct=%r)\n"
            "print('trailing-year zero-return proportion: median %%.2f%%%%  max %%.2f%%%% of days'\n"
            "      %% (R['zp_med_pct'], R['zp_max_pct']))\n"
            "print('long high-zero / short low-zero spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  illiquid book %%+.2f bps vs liquid book %%+.2f bps'\n"
            "      %% (R['hi_bps'], R['lo_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"],
               R["zp_med_pct"], R["zp_max_pct"])
        ),
        md("## 3. Is the machinery even working? A live synthetic control\n\n"
           "We plant the premium in a seeded toy world (`edge>0`, where often-zero names "
           "really do earn more) and check the detector recovers it — and that it stays "
           "*silent* on the null (`edge=0`, zero-days present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from zero_return import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=811, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.012, seed=811, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 4. The honest verdict — no premium here\n\n"
           f"On this liquid mega-cap tape the long-high-zero / short-low-zero spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"statistically indistinguishable from zero (and if anything faintly the wrong "
           f"way). A 1,000-permutation null centres at zero (sd {R['placebo_sd']:.2f} bps) "
           f"and the observed value sits only ~{R['placebo_sigma_left']:.1f}σ into the tail. "
           "The illiquidity premium is a small-and-illiquid-stock phenomenon; on 50 "
           "mega-caps the proxy is near-degenerate and there is nothing to harvest. "
           "**Signal: None**, **Tradability: Mirage** (the book loses money gross, and the "
           "long leg's real trading costs dwarf any edge)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 811 — Zero-Return Illiquidity — the teardown\n\n"
           "The signal degeneracy, the per-leg splits, the Newey-West spread *t*, the "
           "pooled Welch book test, the 1,000-permutation placebo, the two-era robustness "
           "cut, the costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## Signal degeneracy — the reason to expect None\n\n"
           "Trailing-252-day zero-return proportion across the 50 names (final row)."),
        code(
            "print(f\"zero-return proportion: min {R['zp_min_pct']:.2f}%  median {R['zp_med_pct']:.2f}%  \"\n"
            "      f\"max {R['zp_max_pct']:.2f}%  of days\")\n"
            "print(f\"long (top-30%) book carries only ~{R['long_zp_pct']:.2f}% zero-days on average\")\n"
            "print('half the mega-caps sit at exactly 0.00% -> the short leg is a tie-broken pile')"
        ),
        md("## The headline — long-high-zero / short-low-zero spread\n\n"
           "Daily equal-weight top-30% minus bottom-30% zero-proportion spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : illiquid {R['hi_bps']:+.2f} vs liquid {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.3f} \"\n"
            "      f\"(~{R['placebo_sigma_left']:.1f} sigma into the left tail)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('neither era clears |t| >= 2')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per day on the long-short book; short pays 50 bps/yr "
           "borrow. (And the long leg is the least-liquid names — the charged cost is a floor.)"),
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
            "from zero_return import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=811+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.012, seed=811, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.012): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The Lesmond-Ogden-Trzcinka zero-return illiquidity premium does"
           f" **not** appear on 50 liquid US mega-caps: the long-high-zero / short-low-zero spread"
           f" is **{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**, |t| < 2), "
           f"insignificant and weakly wrong-signed, not stable across eras (*t* = "
           f"{R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). The proxy is near-degenerate here "
           f"(median trailing zero-proportion {R['zp_med_pct']:.2f}%) — the effect lives in small, "
           f"tick-priced names, not mega-caps. The 20-seed synthetic control recovers a *planted* "
           f"premium cleanly (*t* = {R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls), so "
           f"the flat real result is a true absence, not a broken sort.\n"
           f"- **Tradability — Mirage.** The book loses money gross ({R['spread_bps']:+.2f} bps/day)"
           f" and worse net (**{R['timer_1_net']:+.2f} bps/day** at 1 bp, *t* = {R['timer_1_t']:+.2f}; "
           f"**{R['timer_5_net']:+.2f}** at 5 bps), and the long leg's real costs are far above the "
           f"optimistic floor charged."),
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
